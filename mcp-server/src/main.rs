//! mev-mcp — a fast local MCP server over the MEV research corpus.
//!
//! Protocol: MCP over stdio (JSON-RPC 2.0, line-delimited).
//! Tools: search, get_paper, list_papers, list_topics, cite.

use anyhow::{Context, Result};
use clap::Parser;
use once_cell::sync::OnceCell;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::document::Value as _;
use tantivy::snippet::SnippetGenerator;
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Parser, Debug)]
#[command(name = "mev-mcp")]
struct Args {
    /// Path to the corpus directory (containing index.sqlite, tantivy/, text/)
    #[arg(long)]
    corpus: PathBuf,
}

const SEARCH_POOL: usize = 60;
const RERANK_POOL: usize = 25;
const DEFAULT_K: usize = 8;
const RRF_K: f32 = 60.0;
const SNIPPET_MAX_CHARS: usize = 600;
const DEFAULT_PER_PAPER_CAP: usize = 2;

/// Query expansion: map common MEV abbreviations to a "term1 OR term2 ..." form
/// so BM25 picks up either side. Embedding side gets the appended aliases too.
fn expand_query(q: &str) -> String {
    let lc = q.to_lowercase();
    let mut additions: Vec<&'static str> = Vec::new();
    let aliases: &[(&[&str], &[&str])] = &[
        (&["mev"], &["maximal extractable value", "miner extractable value"]),
        (&["maximal extractable value", "miner extractable value"], &["MEV"]),
        (&["lvr"], &["loss-versus-rebalancing", "loss versus rebalancing"]),
        (&["loss versus rebalancing", "loss-versus-rebalancing"], &["LVR"]),
        (&["pbs"], &["proposer-builder separation", "proposer builder separation"]),
        (&["proposer-builder separation", "proposer builder separation"], &["PBS", "mev-boost"]),
        (&["amm"], &["automated market maker", "constant function market maker", "CFMM"]),
        (&["cfmm"], &["AMM", "constant function market maker"]),
        (&["constant function market maker"], &["AMM", "CFMM"]),
        (&["intent", "intents"], &["solver", "order flow auction"]),
        (&["zk-rollup", "zk rollup"], &["validity rollup", "zkrollup"]),
        (&["optimistic rollup"], &["fraud proof", "challenge period"]),
        (&["sandwich"], &["front-running", "back-running"]),
        (&["tfm"], &["transaction fee mechanism", "EIP-1559"]),
        (&["eip-1559", "eip 1559"], &["base fee", "transaction fee mechanism"]),
        (&["ofa"], &["order flow auction"]),
        (&["dex"], &["decentralized exchange"]),
    ];
    for (keys, vals) in aliases {
        if keys.iter().any(|k| lc.contains(k)) {
            for v in *vals {
                if !lc.contains(&v.to_lowercase()) {
                    additions.push(v);
                }
            }
        }
    }
    if additions.is_empty() {
        return q.to_string();
    }
    let mut out = q.to_string();
    out.push(' ');
    out.push_str(&additions.join(" "));
    out
}

// ===========================================================================
//  Embedding model — loaded once on first search call.
// ===========================================================================

struct EmbedModel {
    model: Mutex<fastembed::TextEmbedding>,
}

impl EmbedModel {
    fn new() -> Result<Self> {
        let model = fastembed::TextEmbedding::try_new(
            fastembed::TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(false),
        )?;
        Ok(Self { model: Mutex::new(model) })
    }

    fn embed_query(&self, q: &str) -> Result<Vec<f32>> {
        let prefixed = format!("query: {}", q);
        let mut model = self.model.lock().unwrap();
        let vecs = model.embed(vec![prefixed], None)?;
        let mut v = vecs.into_iter().next().context("empty embedding result")?;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }
}

static EMBED: OnceCell<EmbedModel> = OnceCell::new();

fn embed_model() -> Result<&'static EmbedModel> {
    if let Some(m) = EMBED.get() {
        return Ok(m);
    }
    let m = EmbedModel::new()?;
    let _ = EMBED.set(m);
    Ok(EMBED.get().unwrap())
}

// ===========================================================================
//  Cross-encoder re-ranker (BGE-reranker-base) — loaded lazily.
// ===========================================================================

struct RerankModel {
    model: Mutex<fastembed::TextRerank>,
}

impl RerankModel {
    fn new() -> Result<Self> {
        let model = fastembed::TextRerank::try_new(
            fastembed::RerankInitOptions::new(fastembed::RerankerModel::BGERerankerBase)
                .with_show_download_progress(false),
        )?;
        Ok(Self { model: Mutex::new(model) })
    }

    fn rerank_scores(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let mut m = self.model.lock().unwrap();
        let results = m.rerank(query.to_string(), docs.to_vec(), false, Some(16))?;
        // results carry their original index; map back to input order.
        let mut scores = vec![0.0_f32; docs.len()];
        for r in results {
            if r.index < scores.len() {
                scores[r.index] = r.score;
            }
        }
        Ok(scores)
    }
}

static RERANK: OnceCell<RerankModel> = OnceCell::new();

fn rerank_model() -> Result<&'static RerankModel> {
    if let Some(m) = RERANK.get() {
        return Ok(m);
    }
    let m = RerankModel::new()?;
    let _ = RERANK.set(m);
    Ok(RERANK.get().unwrap())
}

// ===========================================================================
//  Corpus state — opened once at startup, shared across requests.
// ===========================================================================

fn register_sqlite_vec_extension() {
    // sqlite-vec is statically linked; register it as an auto-extension so
    // every Connection::open in this process gets it loaded automatically.
    type SqliteExtInit = unsafe extern "C" fn(
        *mut rusqlite::ffi::sqlite3,
        *mut *const std::os::raw::c_char,
        *const rusqlite::ffi::sqlite3_api_routines,
    ) -> std::os::raw::c_int;
    unsafe {
        let init_fn: SqliteExtInit =
            std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(init_fn));
    }
}

struct Corpus {
    db: Mutex<Connection>,
    tantivy_index: Index,
    tantivy_reader: IndexReader,
    text_dir: PathBuf,
    field_text: tantivy::schema::Field,
    field_paper: tantivy::schema::Field,
    field_id: tantivy::schema::Field,
    field_section: tantivy::schema::Field,
    field_title: tantivy::schema::Field,
}

impl Corpus {
    fn open(corpus_dir: &PathBuf) -> Result<Self> {
        let sqlite_path = corpus_dir.join("index.sqlite");
        let tantivy_dir = corpus_dir.join("tantivy");
        let text_dir = corpus_dir.join("text");

        let db = Connection::open_with_flags(
            &sqlite_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        // Confirm vec extension is live.
        let v: String = db.query_row("SELECT vec_version()", [], |r| r.get(0))?;
        tracing::info!("sqlite-vec {} loaded", v);

        let index = Index::open_in_dir(&tantivy_dir)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        let schema = index.schema();
        let field_text = schema.get_field("text")?;
        let field_paper = schema.get_field("paper_id")?;
        let field_id = schema.get_field("id")?;
        let field_section = schema.get_field("section")?;
        let field_title = schema.get_field("title")?;

        Ok(Self {
            db: Mutex::new(db),
            tantivy_index: index,
            tantivy_reader: reader,
            text_dir,
            field_text,
            field_paper,
            field_id,
            field_section,
            field_title,
        })
    }
}

// ===========================================================================
//  Search — hybrid: tantivy BM25 + sqlite-vec cosine, fused with RRF.
// ===========================================================================

#[derive(Debug, Clone, Serialize)]
struct SearchHit {
    chunk_id: String,
    paper_id: String,
    title: String,
    section: String,
    score: f32,
    snippet: String,
    bm25: Option<f32>,
    cosine: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SearchArgs {
    query: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_mode")]
    mode: String, // "hybrid" | "lex" | "sem"
    /// Max chunks returned from the same paper. Default = 2. Pass 0 to disable.
    #[serde(default = "default_per_paper_cap")]
    per_paper_cap: usize,
    /// Apply BGE cross-encoder rerank on the fused top-N before returning.
    #[serde(default = "default_rerank")]
    rerank: bool,
    /// Apply MEV alias query expansion (default on).
    #[serde(default = "default_expand")]
    expand: bool,
}

fn default_k() -> usize { DEFAULT_K }
fn default_mode() -> String { "hybrid".into() }
fn default_per_paper_cap() -> usize { DEFAULT_PER_PAPER_CAP }
fn default_rerank() -> bool { false }
fn default_expand() -> bool { true }

struct LexHit {
    chunk_id: String,
    score: f32,
    snippet: Option<String>,
}

fn lexical_search(corpus: &Corpus, query: &str, n: usize) -> Result<Vec<LexHit>> {
    let searcher = corpus.tantivy_reader.searcher();
    let mut parser = QueryParser::for_index(
        &corpus.tantivy_index,
        vec![corpus.field_text, corpus.field_title, corpus.field_section],
    );
    parser.set_field_boost(corpus.field_title, 2.0);
    let (q, _errors) = parser.parse_query_lenient(query);
    let top: Vec<(f32, tantivy::DocAddress)> =
        searcher.search(&q, &TopDocs::with_limit(n).order_by_score())?;
    let mut snippet_gen = SnippetGenerator::create(&searcher, &*q, corpus.field_text).ok();
    if let Some(ref mut g) = snippet_gen {
        g.set_max_num_chars(SNIPPET_MAX_CHARS);
    }
    let mut out = Vec::with_capacity(top.len());
    for (score, addr) in top {
        let doc: TantivyDocument = searcher.doc(addr)?;
        let cid = doc
            .get_first(corpus.field_id)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if cid.is_empty() {
            continue;
        }
        let snippet: Option<String> = snippet_gen
            .as_ref()
            .map(|g: &SnippetGenerator| g.snippet_from_doc(&doc).fragment().to_string())
            .filter(|s: &String| !s.trim().is_empty());
        out.push(LexHit { chunk_id: cid, score, snippet });
    }
    Ok(out)
}

fn vector_search(corpus: &Corpus, query: &str, n: usize) -> Result<Vec<(String, f32)>> {
    let model = embed_model()?;
    let qvec = model.embed_query(query)?;
    let mut bytes = Vec::with_capacity(qvec.len() * 4);
    for x in &qvec {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let db = corpus.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT c.chunk_id, v.distance \
         FROM chunks_vec v JOIN chunks c ON c.rowid = v.rowid \
         WHERE v.embedding MATCH ?1 AND k = ?2 \
         ORDER BY v.distance",
    )?;
    let rows = stmt.query_map(params![bytes, n as i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
    })?;
    let mut out = Vec::new();
    for r in rows {
        let (cid, dist) = r?;
        // cosine distance ∈ [0,2]; convert to similarity = 1 - dist/2
        let sim = 1.0 - dist / 2.0;
        out.push((cid, sim));
    }
    Ok(out)
}

fn rrf_fuse(lex_ids: &[String], vec_ids: &[String]) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (rank, cid) in lex_ids.iter().enumerate() {
        *scores.entry(cid.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, cid) in vec_ids.iter().enumerate() {
        *scores.entry(cid.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    let mut v: Vec<(String, f32)> = scores.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn hit_snippet(text: &str) -> String {
    if text.chars().count() <= SNIPPET_MAX_CHARS {
        return text.to_string();
    }
    let mut end = SNIPPET_MAX_CHARS.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

fn pick_snippet(text: &str, lex_snippet: Option<&str>) -> String {
    // Prefer the lexical match-centered snippet when present; otherwise
    // fall back to the head of the chunk.
    if let Some(s) = lex_snippet {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    hit_snippet(text)
}

fn fetch_chunks(
    corpus: &Corpus,
    chunk_ids: &[String],
    lex_scores: &HashMap<String, f32>,
    lex_snippets: &HashMap<String, String>,
    vec_sim: &HashMap<String, f32>,
    final_scores: &HashMap<String, f32>,
) -> Result<Vec<SearchHit>> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = chunk_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT c.chunk_id, c.paper_id, c.section, c.text, p.title \
         FROM chunks c JOIN papers p ON p.id = c.paper_id \
         WHERE c.chunk_id IN ({})",
        placeholders
    );
    let db = corpus.db.lock().unwrap();
    let mut stmt = db.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        chunk_ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(&*params, |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut by_id: HashMap<String, SearchHit> = HashMap::new();
    for r in rows {
        let (cid, pid, section, text, title) = r?;
        let snippet = pick_snippet(&text, lex_snippets.get(&cid).map(|s| s.as_str()));
        by_id.insert(
            cid.clone(),
            SearchHit {
                chunk_id: cid.clone(),
                paper_id: pid,
                title,
                section,
                score: *final_scores.get(&cid).unwrap_or(&0.0),
                snippet,
                bm25: lex_scores.get(&cid).copied(),
                cosine: vec_sim.get(&cid).copied(),
            },
        );
    }
    let mut out = Vec::with_capacity(chunk_ids.len());
    for cid in chunk_ids {
        if let Some(h) = by_id.remove(cid) {
            out.push(h);
        }
    }
    Ok(out)
}

fn do_search(corpus: &Corpus, args: SearchArgs) -> Result<Value> {
    let k = args.k.max(1).min(50);
    let effective_query = if args.expand { expand_query(&args.query) } else { args.query.clone() };

    let lex_hits = if args.mode != "sem" {
        lexical_search(corpus, &effective_query, SEARCH_POOL)?
    } else {
        Vec::new()
    };
    let vec_hits = if args.mode != "lex" {
        match vector_search(corpus, &effective_query, SEARCH_POOL) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("vector search failed: {e:?}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let lex_ids: Vec<String> = lex_hits.iter().map(|h| h.chunk_id.clone()).collect();
    let vec_ids: Vec<String> = vec_hits.iter().map(|(c, _)| c.clone()).collect();

    let mut fused: Vec<(String, f32)> = match args.mode.as_str() {
        "lex" => lex_hits.iter().map(|h| (h.chunk_id.clone(), h.score)).collect(),
        "sem" => vec_hits.clone(),
        _ => rrf_fuse(&lex_ids, &vec_ids),
    };

    // Optional re-ranking pass on the top-N fused.
    if args.rerank && !fused.is_empty() {
        let pool: Vec<String> = fused.iter().take(RERANK_POOL).map(|(c, _)| c.clone()).collect();
        let docs = chunk_texts(corpus, &pool)?;
        let rr = rerank_model();
        if let Ok(rr) = rr {
            let inputs: Vec<String> = docs.iter().map(|(_, t)| t.clone()).collect();
            match rr.rerank_scores(&args.query, &inputs) {
                Ok(scores) => {
                    let mut ranked: Vec<(String, f32)> = pool
                        .iter()
                        .zip(scores.into_iter())
                        .map(|(c, s)| (c.clone(), s))
                        .collect();
                    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    // Append any fused items not in pool (preserve recall tail).
                    let in_pool: HashSet<String> = pool.iter().cloned().collect();
                    for (c, s) in fused.iter() {
                        if !in_pool.contains(c) {
                            ranked.push((c.clone(), *s));
                        }
                    }
                    fused = ranked;
                }
                Err(e) => tracing::warn!("rerank failed: {e:?}"),
            }
        }
    }

    let lex_scores: HashMap<String, f32> =
        lex_hits.iter().map(|h| (h.chunk_id.clone(), h.score)).collect();
    let lex_snippets: HashMap<String, String> = lex_hits
        .iter()
        .filter_map(|h| h.snippet.as_ref().map(|s| (h.chunk_id.clone(), s.clone())))
        .collect();
    let vec_map: HashMap<String, f32> = vec_hits.into_iter().collect();
    let final_map: HashMap<String, f32> =
        fused.iter().map(|(c, s)| (c.clone(), *s)).collect();

    // Apply per-paper diversity cap. paper_id is encoded as the chunk_id
    // prefix before "::N" — but easier and safer is to look up from SQLite.
    // Build a chunk_id → paper_id map for the fused set (small).
    let paper_lookup = paper_id_lookup(corpus, &fused.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>())?;

    let cap = args.per_paper_cap;
    let mut seen_per_paper: HashMap<String, usize> = HashMap::new();
    let mut ids: Vec<String> = Vec::with_capacity(k);
    for (cid, _) in &fused {
        if ids.len() >= k {
            break;
        }
        let pid = paper_lookup.get(cid).cloned().unwrap_or_default();
        if cap > 0 && !pid.is_empty() {
            let cnt = seen_per_paper.entry(pid.clone()).or_insert(0);
            if *cnt >= cap {
                continue;
            }
            *cnt += 1;
        }
        ids.push(cid.clone());
    }

    let hits = fetch_chunks(corpus, &ids, &lex_scores, &lex_snippets, &vec_map, &final_map)?;
    Ok(json!({ "hits": hits, "k": k, "mode": args.mode, "per_paper_cap": cap }))
}

fn chunk_texts(corpus: &Corpus, chunk_ids: &[String]) -> Result<Vec<(String, String)>> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = chunk_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({})",
        placeholders
    );
    let db = corpus.db.lock().unwrap();
    let mut stmt = db.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        chunk_ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(&*params, |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut by_id: HashMap<String, String> = HashMap::new();
    for r in rows {
        let (c, t) = r?;
        by_id.insert(c, t);
    }
    let mut out: Vec<(String, String)> = Vec::with_capacity(chunk_ids.len());
    for cid in chunk_ids {
        if let Some(t) = by_id.remove(cid) {
            out.push((cid.clone(), t));
        }
    }
    Ok(out)
}

fn paper_id_lookup(corpus: &Corpus, chunk_ids: &[String]) -> Result<HashMap<String, String>> {
    if chunk_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = chunk_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT chunk_id, paper_id FROM chunks WHERE chunk_id IN ({})",
        placeholders
    );
    let db = corpus.db.lock().unwrap();
    let mut stmt = db.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        chunk_ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let mut out = HashMap::with_capacity(chunk_ids.len());
    let rows = stmt.query_map(&*params, |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for r in rows {
        let (c, p) = r?;
        out.insert(c, p);
    }
    Ok(out)
}

// ===========================================================================
//  Other tools
// ===========================================================================

#[derive(Debug, Deserialize)]
struct GetPaperArgs {
    paper_id: String,
    #[serde(default)]
    sections: Option<Vec<String>>,
}

fn do_get_paper(corpus: &Corpus, args: GetPaperArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let meta: Option<(String, String, String, String, String, String, i64, String)> = db
        .query_row(
            "SELECT title, authors, topics_json, release_date, pdf_url, sha256, n_pages, abstract \
             FROM papers WHERE id = ?1",
            params![args.paper_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            },
        )
        .ok();
    drop(db);
    let Some((title, authors, topics_json, date, url, sha, n_pages, abstract_)) = meta else {
        return Ok(json!({"error": "paper_id not found"}));
    };

    let md_path = corpus.text_dir.join(format!("{}.md", args.paper_id));
    let md = std::fs::read_to_string(&md_path).unwrap_or_default();

    let body: String = match args.sections {
        Some(wanted) if !wanted.is_empty() => extract_sections(&md, &wanted),
        _ => md,
    };

    Ok(json!({
        "id": args.paper_id,
        "title": title,
        "authors": authors,
        "topics": serde_json::from_str::<Value>(&topics_json).unwrap_or(json!([])),
        "release_date": date,
        "pdf_url": url,
        "sha256": sha,
        "n_pages": n_pages,
        "abstract": abstract_,
        "markdown": body,
    }))
}

fn extract_sections(md: &str, wanted: &[String]) -> String {
    let wanted_lc: HashSet<String> = wanted.iter().map(|s| s.to_lowercase()).collect();
    let mut out = String::new();
    let mut keep = false;
    for line in md.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            let h = rest.trim().to_lowercase();
            keep = wanted_lc.iter().any(|w| h.contains(w));
            if keep {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }
        if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct ListPapersArgs {
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    since: Option<String>,
    #[serde(default = "list_default_limit")]
    limit: usize,
}

fn list_default_limit() -> usize { 50 }

fn do_list_papers(corpus: &Corpus, args: ListPapersArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let mut sql =
        "SELECT id, title, authors, topics_json, release_date FROM papers WHERE 1=1".to_string();
    let mut bound: Vec<String> = Vec::new();
    if let Some(t) = &args.topic {
        sql.push_str(" AND topics_json LIKE ?");
        bound.push(format!("%\"{}\"%", t));
    }
    if let Some(d) = &args.since {
        sql.push_str(" AND release_date >= ?");
        bound.push(d.clone());
    }
    sql.push_str(" ORDER BY release_date DESC LIMIT ?");
    bound.push(args.limit.to_string());
    let mut stmt = db.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> =
        bound.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(&*params, |row| {
        Ok(json!({
            "id": row.get::<_, String>(0)?,
            "title": row.get::<_, String>(1)?,
            "authors": row.get::<_, String>(2)?,
            "topics": serde_json::from_str::<Value>(&row.get::<_, String>(3)?).unwrap_or(json!([])),
            "date": row.get::<_, String>(4)?,
        }))
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    let n = out.len();
    Ok(json!({ "papers": out, "count": n }))
}

fn do_list_topics(corpus: &Corpus) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let mut stmt = db.prepare("SELECT topics_json FROM papers")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut counts: HashMap<String, i64> = HashMap::new();
    for r in rows {
        let s = r?;
        if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&s) {
            for v in arr {
                if let Value::String(t) = v {
                    *counts.entry(t).or_insert(0) += 1;
                }
            }
        }
    }
    let mut list: Vec<(String, i64)> = counts.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1));
    let arr: Vec<Value> = list
        .into_iter()
        .map(|(t, c)| json!({ "topic": t, "count": c }))
        .collect();
    Ok(json!({ "topics": arr }))
}

#[derive(Debug, Deserialize)]
struct SummarizeArgs {
    paper_id: String,
}

fn do_summarize_paper(corpus: &Corpus, args: SummarizeArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let row: Option<(String, String, String, String, String, String, i64, String)> = db
        .query_row(
            "SELECT title, authors, topics_json, release_date, pdf_url, sha256, n_pages, abstract \
             FROM papers WHERE id = ?1",
            params![args.paper_id],
            |row| Ok((
                row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?
            )),
        )
        .ok();
    let Some((title, authors, topics_json, date, url, _sha, n_pages, abstract_)) = row else {
        return Ok(json!({"error": "paper_id not found"}));
    };
    // Pull a small set of "structural" chunks: prefer Intro / Conclusion / Discussion
    // sections, plus the first 1–2 chunks (which usually contain title page + abstract).
    let mut stmt = db.prepare(
        "SELECT chunk_id, section, text FROM chunks WHERE paper_id = ?1 ORDER BY rowid"
    )?;
    let rows = stmt.query_map(params![args.paper_id], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
    })?;
    let mut chunks: Vec<(String, String, String)> = Vec::new();
    for r in rows {
        chunks.push(r?);
    }
    drop(stmt);
    drop(db);
    let mut picks: Vec<&(String, String, String)> = Vec::new();
    let key_terms = ["introduction", "conclusion", "discussion", "summary", "results", "contribution"];
    let mut seen_sections: HashSet<String> = HashSet::new();
    // First, take up to 2 head chunks for the abstract / intro paragraph.
    for c in chunks.iter().take(2) {
        picks.push(c);
        seen_sections.insert(c.1.to_lowercase());
    }
    for term in &key_terms {
        for c in &chunks {
            let s = c.1.to_lowercase();
            if !seen_sections.contains(&s) && s.contains(term) {
                picks.push(c);
                seen_sections.insert(s);
                break;
            }
        }
    }
    // Cap total chars at ~5000 to keep response compact.
    const MAX_CHARS: usize = 6000;
    let mut total = 0usize;
    let key_excerpts: Vec<Value> = picks
        .iter()
        .filter_map(|(cid, sec, text)| {
            if total >= MAX_CHARS {
                return None;
            }
            let remaining = MAX_CHARS - total;
            let snippet = if text.chars().count() > remaining {
                let mut end = remaining.min(text.len());
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}…", &text[..end])
            } else {
                text.clone()
            };
            total += snippet.chars().count();
            Some(json!({
                "chunk_id": cid,
                "section": sec,
                "text": snippet,
            }))
        })
        .collect();
    Ok(json!({
        "id": args.paper_id,
        "title": title,
        "authors": authors,
        "topics": serde_json::from_str::<Value>(&topics_json).unwrap_or(json!([])),
        "release_date": date,
        "pdf_url": url,
        "n_pages": n_pages,
        "abstract": abstract_,
        "key_excerpts": key_excerpts,
        "n_chunks": chunks.len(),
    }))
}

#[derive(Debug, Deserialize)]
struct FindRelatedArgs {
    paper_id: String,
    #[serde(default = "default_related_k")]
    k: usize,
}
fn default_related_k() -> usize { 10 }

fn do_find_related(corpus: &Corpus, args: FindRelatedArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let paper_rowid: Option<i64> = db
        .query_row(
            "SELECT paper_rowid FROM paper_rowid_map WHERE paper_id = ?1",
            params![args.paper_id],
            |r| r.get::<_, i64>(0),
        )
        .ok();
    let Some(paper_rowid) = paper_rowid else {
        return Ok(json!({"error": "paper_id not found in paper_rowid_map"}));
    };
    // Fetch the paper's centroid embedding.
    let emb: Vec<u8> = db.query_row(
        "SELECT embedding FROM papers_vec WHERE paper_rowid = ?1",
        params![paper_rowid],
        |r| r.get::<_, Vec<u8>>(0),
    )?;
    let k = args.k.max(1).min(30);
    // KNN search excluding the source paper itself.
    let mut stmt = db.prepare(
        "SELECT pv.paper_rowid, pv.distance, prm.paper_id, p.title, p.release_date \
         FROM papers_vec pv JOIN paper_rowid_map prm ON prm.paper_rowid = pv.paper_rowid \
         JOIN papers p ON p.id = prm.paper_id \
         WHERE pv.embedding MATCH ?1 AND pv.k = ?2 \
         ORDER BY pv.distance"
    )?;
    let rows = stmt.query_map(params![emb, (k + 1) as i64], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, f64>(1)? as f32,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    })?;
    let mut out: Vec<Value> = Vec::new();
    for r in rows {
        let (rid, dist, pid, title, date) = r?;
        if rid == paper_rowid {
            continue;
        }
        let sim = 1.0 - dist / 2.0;
        out.push(json!({
            "paper_id": pid,
            "title": title,
            "date": date,
            "similarity": sim,
        }));
        if out.len() >= k {
            break;
        }
    }
    Ok(json!({ "source": args.paper_id, "related": out }))
}

#[derive(Debug, Deserialize)]
struct CitationsArgs {
    paper_id: String,
    /// "cites" — papers this one cites; "cited_by" — papers that cite this one. Default: both.
    #[serde(default)]
    direction: Option<String>,
}

fn do_citations(corpus: &Corpus, args: CitationsArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let mut cites = json!([]);
    let mut cited_by = json!([]);
    let direction = args.direction.as_deref().unwrap_or("both");
    if direction == "cites" || direction == "both" {
        let mut stmt = db.prepare(
            "SELECT c.dst_paper, p.title, SUM(c.src_count) AS n FROM citations c \
             JOIN papers p ON p.id = c.dst_paper WHERE c.src_paper = ?1 \
             GROUP BY c.dst_paper ORDER BY n DESC"
        )?;
        let rows = stmt.query_map(params![args.paper_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;
        let mut v = Vec::new();
        for r in rows {
            let (pid, title, n) = r?;
            v.push(json!({"paper_id": pid, "title": title, "count": n}));
        }
        cites = Value::Array(v);
    }
    if direction == "cited_by" || direction == "both" {
        let mut stmt = db.prepare(
            "SELECT c.src_paper, p.title, SUM(c.src_count) AS n FROM citations c \
             JOIN papers p ON p.id = c.src_paper WHERE c.dst_paper = ?1 \
             GROUP BY c.src_paper ORDER BY n DESC"
        )?;
        let rows = stmt.query_map(params![args.paper_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;
        let mut v = Vec::new();
        for r in rows {
            let (pid, title, n) = r?;
            v.push(json!({"paper_id": pid, "title": title, "count": n}));
        }
        cited_by = Value::Array(v);
    }
    Ok(json!({
        "paper_id": args.paper_id,
        "cites": cites,
        "cited_by": cited_by,
    }))
}

#[derive(Debug, Deserialize)]
struct CiteArgs {
    paper_id: String,
}

fn do_cite(corpus: &Corpus, args: CiteArgs) -> Result<Value> {
    let db = corpus.db.lock().unwrap();
    let row: Option<(String, String, String)> = db
        .query_row(
            "SELECT title, authors, release_date FROM papers WHERE id = ?1",
            params![args.paper_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .ok();
    let Some((title, authors, date)) = row else {
        return Ok(json!({"error": "paper_id not found"}));
    };
    let year = date.split('-').next().unwrap_or("n.d.");
    let first_author = authors.split(',').next().unwrap_or("Anonymous").trim();
    let key = format!(
        "{}{}",
        first_author
            .split_whitespace()
            .last()
            .unwrap_or("anon")
            .to_lowercase()
            .replace(|c: char| !c.is_ascii_alphanumeric(), ""),
        year
    );
    let bibtex = format!(
        "@misc{{{key},\n  title  = {{{}}},\n  author = {{{}}},\n  year   = {{{}}},\n  note   = {{paper_id: {}}}\n}}",
        title, authors, year, args.paper_id
    );
    Ok(json!({ "bibtex": bibtex, "key": key }))
}

// ===========================================================================
//  MCP protocol — JSON-RPC 2.0 over stdio.
// ===========================================================================

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

fn make_response(id: Option<Value>, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}
fn make_error(id: Option<Value>, code: i64, msg: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg } })
}

fn tools_descriptor() -> Value {
    json!([
        {
            "name": "search",
            "description": "Hybrid (BM25 + semantic) search across the MEV research corpus. Returns ranked chunks with paper context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language query" },
                    "k": { "type": "integer", "default": 8, "minimum": 1, "maximum": 50 },
                    "mode": { "type": "string", "enum": ["hybrid","lex","sem"], "default": "hybrid" },
                    "per_paper_cap": { "type": "integer", "default": 2, "minimum": 0, "description": "Max chunks per paper in the output (0 = unlimited)" },
                    "rerank": { "type": "boolean", "default": false, "description": "Apply BGE cross-encoder rerank on the top-25 fused hits. Higher precision on ambiguous queries but adds ~3-6s on CPU." },
                    "expand": { "type": "boolean", "default": true, "description": "Apply MEV alias query expansion (MEV/LVR/PBS/AMM/...)." }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_paper",
            "description": "Fetch full or section-filtered markdown of a paper plus metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_id": { "type": "string" },
                    "sections": { "type": "array", "items": {"type":"string"}, "description": "Substring match on section headings (case-insensitive)." }
                },
                "required": ["paper_id"]
            }
        },
        {
            "name": "list_papers",
            "description": "List papers, optionally filtered by topic/date.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "topic": { "type": "string" },
                    "since": { "type": "string", "description": "ISO date YYYY-MM-DD" },
                    "limit": { "type": "integer", "default": 50 }
                }
            }
        },
        {
            "name": "list_topics",
            "description": "Topic facets with counts.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "cite",
            "description": "Generate a BibTeX entry for a paper_id.",
            "inputSchema": {
                "type": "object",
                "properties": { "paper_id": { "type": "string" } },
                "required": ["paper_id"]
            }
        },
        {
            "name": "summarize_paper",
            "description": "Structured TL;DR for a paper: metadata + curated key excerpts from intro/conclusion/discussion sections.",
            "inputSchema": {
                "type": "object",
                "properties": { "paper_id": { "type": "string" } },
                "required": ["paper_id"]
            }
        },
        {
            "name": "find_related",
            "description": "Find papers semantically related to the given paper (kNN over per-paper centroid embeddings).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_id": { "type": "string" },
                    "k": { "type": "integer", "default": 10, "minimum": 1, "maximum": 30 }
                },
                "required": ["paper_id"]
            }
        },
        {
            "name": "citations",
            "description": "Citation graph queries (extracted via arXiv-id / DOI matching across the corpus).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paper_id": { "type": "string" },
                    "direction": { "type": "string", "enum": ["cites", "cited_by", "both"], "default": "both" }
                },
                "required": ["paper_id"]
            }
        }
    ])
}

async fn handle(corpus: Arc<Corpus>, req: RpcRequest) -> Value {
    let id = req.id.clone();
    match req.method.as_str() {
        "initialize" => make_response(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "mev-mcp", "version": env!("CARGO_PKG_VERSION") }
            }),
        ),
        "tools/list" => make_response(id, json!({ "tools": tools_descriptor() })),
        "tools/call" => {
            let name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            let result: Result<Value> = tokio::task::spawn_blocking({
                let corpus = corpus.clone();
                let name = name.to_string();
                move || dispatch(&corpus, &name, args)
            })
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("join error: {e}")));
            match result {
                Ok(payload) => make_response(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": serde_json::to_string(&payload).unwrap_or_default() }],
                        "isError": false
                    }),
                ),
                Err(e) => make_response(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": format!("error: {e:#}") }],
                        "isError": true
                    }),
                ),
            }
        }
        "notifications/initialized" | "notifications/cancelled" => Value::Null,
        "ping" => make_response(id, json!({})),
        _ => make_error(id, -32601, "method not found"),
    }
}

fn dispatch(corpus: &Corpus, name: &str, args: Value) -> Result<Value> {
    match name {
        "search" => do_search(corpus, serde_json::from_value(args)?),
        "get_paper" => do_get_paper(corpus, serde_json::from_value(args)?),
        "list_papers" => do_list_papers(corpus, serde_json::from_value(args)?),
        "list_topics" => do_list_topics(corpus),
        "cite" => do_cite(corpus, serde_json::from_value(args)?),
        "summarize_paper" => do_summarize_paper(corpus, serde_json::from_value(args)?),
        "find_related" => do_find_related(corpus, serde_json::from_value(args)?),
        "citations" => do_citations(corpus, serde_json::from_value(args)?),
        _ => Err(anyhow::anyhow!("unknown tool: {name}")),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    register_sqlite_vec_extension();

    let args = Args::parse();
    let corpus = Arc::new(Corpus::open(&args.corpus)?);
    tracing::info!("corpus opened: {}", args.corpus.display());

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: RpcRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let resp = make_error(None, -32700, &format!("parse error: {e}"));
                let mut s = serde_json::to_string(&resp)?;
                s.push('\n');
                stdout.write_all(s.as_bytes()).await?;
                stdout.flush().await?;
                continue;
            }
        };
        if req.jsonrpc != "2.0" {
            continue;
        }
        let is_notification = req.id.is_none();
        let resp = handle(corpus.clone(), req).await;
        if !is_notification && !resp.is_null() {
            let mut s = serde_json::to_string(&resp)?;
            s.push('\n');
            stdout.write_all(s.as_bytes()).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}
