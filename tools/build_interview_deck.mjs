import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  FileBlob,
  Presentation,
  PresentationFile,
} from "@oai/artifact-tool";

const workspaceDir = process.cwd();
const SKILL_DIR = "C:/Users/asevi/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const RUNTIME_PYTHON = "C:/Users/asevi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const TMP_DIR = path.join(workspaceDir, ".pptx-build");
const FINAL_PPTX = path.join(
  workspaceDir,
  "deliverables",
  "Allianz_Claims_Copilot_Interview_Presentation_v2.pptx",
);

const {
  finalizePresentation,
  makeNativeBulletParagraphs,
  resolvePresentationFont,
} = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const family = resolvePresentationFont();
const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const C = {
  navy: "#071E2E",
  blue: "#005B96",
  cyan: "#35A8E0",
  red: "#D71920",
  ink: "#1D2730",
  gray: "#5D6A73",
  line: "#CBD5DB",
  pale: "#EAF3F8",
  white: "#FFFFFF",
  green: "#198754",
  amber: "#B36B00",
};

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    autoFit: "none",
    verticalAlignment: options.verticalAlignment ?? "middle",
    horizontalAlignment: options.align ?? "left",
  };
  return shape;
}

function addTitle(slide, title, index) {
  addText(slide, title, { left: 72, top: 40, width: 1090, height: 62 }, {
    fontSize: 38,
    bold: true,
    color: C.navy,
  });
  const rule = slide.shapes.add({
    geometry: "rect",
    position: { left: 72, top: 112, width: 82, height: 6 },
    fill: C.red,
    line: { fill: "none", width: 0 },
  });
  rule.name = "Title accent";
  addText(slide, String(index).padStart(2, "0"), {
    left: 1170, top: 48, width: 50, height: 34,
  }, { fontSize: 19, bold: true, color: C.gray, align: "right" });
}

function addBullets(slide, items, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: 18,
    hangingPoints: 8,
    spaceAfterPoints: options.spaceAfter ?? 10,
  });
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 24,
    color: options.color ?? C.ink,
    autoFit: "none",
  };
  return shape;
}

function addLabel(slide, text, left, top, width, color = C.blue) {
  const shape = slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height: 42 },
    fill: color,
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: 18,
    bold: true,
    color: C.white,
    autoFit: "none",
    horizontalAlignment: "center",
    verticalAlignment: "middle",
  };
  return shape;
}

function addNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

// 1. Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 18, height: 720 },
    fill: C.red,
    line: { fill: "none", width: 0 },
  });
  addText(slide, "Agentic RAG for\nMotor Claims Handling", {
    left: 92, top: 148, width: 820, height: 190,
  }, { fontSize: 58, bold: true, color: C.white });
  addText(slide, "CIDE, ASCIDE and CICOS evidence assistant", {
    left: 96, top: 360, width: 790, height: 52,
  }, { fontSize: 26, color: "#C9DCE8" });
  addText(slide, "AI Engineer technical assessment", {
    left: 96, top: 540, width: 540, height: 40,
  }, { fontSize: 21, bold: true, color: C.white });
  addText(slide, "Five day implementation", {
    left: 96, top: 585, width: 420, height: 34,
  }, { fontSize: 19, color: "#C9DCE8" });
  addText(slide, "29 September 2026", {
    left: 930, top: 620, width: 250, height: 30,
  }, { fontSize: 17, color: "#A9BFCC", align: "right" });
  addNotes(slide, [
    "Open with the goal: demonstrate an end-to-end AI engineering approach in five days.",
    "State that this is a candidate project and not an official Allianz system.",
    "Source: supplied GenAI Technical Assessment and CIDE/ASCIDE/CICOS manual.",
  ]);
}

// 2. Brief and constraint
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Assessment brief and delivery constraint", 2);
  addText(slide, "The expected system must answer manual questions and interpret accident narratives with an LLM.", {
    left: 72, top: 145, width: 1130, height: 58,
  }, { fontSize: 26, bold: true, color: C.navy });
  addBullets(slide, [
    "Use the supplied 111-page CIDE, ASCIDE and CICOS manual as the knowledge source.",
    "Explain model and retrieval choices during a live interview.",
    "Identify parties, circumstances and agreement responsibility in accident cases.",
    "Deliver source code, a technical document, a presentation and basic evaluation.",
    "Complete a defensible vertical slice within five days.",
  ], { left: 92, top: 235, width: 1080, height: 350 }, { fontSize: 24, spaceAfter: 13 });
  addText(slide, "Design implication", { left: 92, top: 610, width: 210, height: 34 }, {
    fontSize: 20, bold: true, color: C.red,
  });
  addText(slide, "Traceability and a reliable demo matter more than feature breadth.", {
    left: 300, top: 604, width: 830, height: 42,
  }, { fontSize: 22, bold: true, color: C.ink });
  addNotes(slide, [
    "Explain that the repository began with only a README, so project setup formed part of the work.",
    "Do not promise production readiness. The five-day target drives a narrow but complete architecture.",
    "Source: GenAI_Interview_Instructions.docx supplied with the assessment.",
  ]);
}

// 3. User and safety boundary
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "User goal and decision boundary", 3);
  addText(slide, "Claims handler", { left: 88, top: 170, width: 250, height: 52 }, {
    fontSize: 32, bold: true, color: C.navy,
  });
  addBullets(slide, [
    "Ask a factual question about the manual",
    "Describe an accident in natural language",
    "Inspect the evidence behind the answer",
  ], { left: 88, top: 245, width: 480, height: 250 }, { fontSize: 24 });
  const divider = slide.shapes.add({
    geometry: "rect",
    position: { left: 620, top: 165, width: 3, height: 390 },
    fill: C.line,
    line: { fill: "none", width: 0 },
  });
  divider.name = "Boundary divider";
  addText(slide, "Human review remains mandatory", {
    left: 690, top: 170, width: 480, height: 52,
  }, { fontSize: 30, bold: true, color: C.red });
  addBullets(slide, [
    "Agreement responsibility is separate from legal liability",
    "No automated coverage, payment or rejection decision",
    "Missing facts and low confidence remain visible",
    "The supplied source dates from 2004",
  ], { left: 690, top: 245, width: 480, height: 285 }, { fontSize: 23 });
  addText(slide, "The assistant supports a decision. It does not own the decision.", {
    left: 150, top: 610, width: 980, height: 46,
  }, { fontSize: 27, bold: true, color: C.navy, align: "center" });
  addNotes(slide, [
    "Use this slide to establish the high-impact action boundary before discussing the model.",
    "The manual governs operational criteria between insurers and must not be presented as current legal advice.",
    "Source: supplied manual, pages 25 to 27 on culpability, plus repository safety policy.",
  ]);
}

// 4. Architecture
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "End-to-end architecture", 4);
  const y = 258;
  const boxes = [
    ["Input", "question or\naccident", 60, C.navy],
    ["Validation", "route and\nlimits", 260, C.blue],
    ["Retrieval", "Chroma and\npage metadata", 460, C.blue],
    ["Generation", "Gemini and\nstructured output", 660, C.blue],
    ["Validation", "citations and\nretry", 860, C.blue],
    ["Response", "evidence or\nfallback", 1060, C.green],
  ];
  const shapes = boxes.map(([heading, detail, left, color]) => {
    const box = slide.shapes.add({
      geometry: "roundRect",
      position: { left, top: y, width: 160, height: 155 },
      fill: color,
      line: { fill: "none", width: 0 },
    });
    box.text = `${heading}\n${detail}`;
    box.text.style = {
      typeface: family,
      fontSize: 20,
      bold: true,
      color: C.white,
      autoFit: "none",
      horizontalAlignment: "center",
      verticalAlignment: "middle",
    };
    return box;
  });
  for (let i = 0; i < shapes.length - 1; i += 1) {
    slide.shapes.connect(shapes[i], shapes[i + 1], {
      kind: "straight",
      fromSide: "right",
      toSide: "left",
      line: { style: "solid", fill: C.gray, width: 2 },
      tail: { type: "arrow", width: "med", length: "med" },
    });
  }
  addText(slide, "One service backs the CLI, FastAPI and Streamlit interfaces", {
    left: 178, top: 500, width: 930, height: 46,
  }, { fontSize: 27, bold: true, color: C.navy, align: "center" });
  addText(slide, "Environment configuration  •  persistent index  •  external API-free unit tests", {
    left: 205, top: 565, width: 875, height: 36,
  }, { fontSize: 20, color: C.gray, align: "center" });
  addNotes(slide, [
    "Walk left to right. Emphasize that each stage has a typed boundary and can be tested independently.",
    "The arrow diagram is editable in PowerPoint.",
    "Source: repository modules under src/allianz_rag.",
  ]);
}

// 5. Ingestion
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Source ingestion preserves page evidence", 5);
  addText(slide, "111", { left: 92, top: 160, width: 190, height: 100 }, {
    fontSize: 70, bold: true, color: C.blue, align: "center",
  });
  addText(slide, "PDF pages", { left: 92, top: 255, width: 190, height: 36 }, {
    fontSize: 22, bold: true, color: C.gray, align: "center",
  });
  addText(slide, "134", { left: 360, top: 160, width: 190, height: 100 }, {
    fontSize: 70, bold: true, color: C.blue, align: "center",
  });
  addText(slide, "stable chunks", { left: 350, top: 255, width: 210, height: 36 }, {
    fontSize: 22, bold: true, color: C.gray, align: "center",
  });
  addText(slide, "62", { left: 625, top: 160, width: 190, height: 100 }, {
    fontSize: 70, bold: true, color: C.blue, align: "center",
  });
  addText(slide, "detected sections", { left: 605, top: 255, width: 230, height: 36 }, {
    fontSize: 22, bold: true, color: C.gray, align: "center",
  });
  addText(slide, "1", { left: 910, top: 160, width: 190, height: 100 }, {
    fontSize: 70, bold: true, color: C.red, align: "center",
  });
  addText(slide, "page per citation", { left: 885, top: 255, width: 240, height: 36 }, {
    fontSize: 22, bold: true, color: C.gray, align: "center",
  });
  addBullets(slide, [
    "Cleaning is deterministic and never guesses missing accented characters.",
    "Chunks do not cross page boundaries.",
    "Source, page, section and SHA-derived chunk ID travel with every vector.",
    "Spanish OCR remains a documented production improvement.",
  ], { left: 150, top: 365, width: 970, height: 240 }, { fontSize: 24, spaceAfter: 12 });
  addNotes(slide, [
    "Explain the damaged text-layer issue: visible pages are correct, while some extracted accents become unknown glyphs.",
    "The measured counts come from the implemented ingestion pipeline.",
    "Source: supplied 111-page PDF and local build output from allianz_rag.ingestion.",
  ]);
}

// 6. Retrieval
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Retrieval favors privacy and traceability", 6);
  addLabel(slide, "Local multilingual embeddings", 82, 175, 350, C.blue);
  addText(slide, "The manual stays local during indexing. Spanish queries and source text share one embedding space.", {
    left: 82, top: 235, width: 360, height: 180,
  }, { fontSize: 24, color: C.ink });
  addLabel(slide, "Persistent Chroma index", 465, 175, 350, C.blue);
  addText(slide, "Cosine-distance HNSW search keeps source, page and section metadata beside each chunk.", {
    left: 465, top: 235, width: 350, height: 180,
  }, { fontSize: 24, color: C.ink });
  addLabel(slide, "Configurable acceptance", 848, 175, 350, C.blue);
  addText(slide, "Top K equals six by default. A maximum distance rejects weak evidence before generation.", {
    left: 848, top: 235, width: 350, height: 180,
  }, { fontSize: 24, color: C.ink });
  addText(slide, "Retrieval quality is measured independently from answer fluency", {
    left: 145, top: 545, width: 990, height: 55,
  }, { fontSize: 31, bold: true, color: C.navy, align: "center" });
  addNotes(slide, [
    "State the trade-off: the local embedding model increases image size and startup time, but avoids sending the manual to an embedding API.",
    "Chroma is suitable for one manual and an interview demo. A managed regional vector store is a production option.",
    "Source: pyproject.toml, config.py and vector_store.py.",
  ]);
}

// 7. Agent graph
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Agent graph uses deterministic control where possible", 7);
  const steps = [
    ["1", "Validate input"],
    ["2", "Classify route"],
    ["3", "Retrieve evidence"],
    ["4", "Generate schema"],
    ["5", "Check citations"],
    ["6", "Retry once or stop"],
  ];
  const nodes = steps.map(([number, label], index) => {
    const left = 80 + index * 195;
    const circle = slide.shapes.add({
      geometry: "ellipse",
      position: { left, top: 205, width: 74, height: 74 },
      fill: index === 5 ? C.green : C.blue,
      line: { fill: "none", width: 0 },
    });
    circle.text = number;
    circle.text.style = {
      typeface: family, fontSize: 28, bold: true, color: C.white,
      autoFit: "none", horizontalAlignment: "center", verticalAlignment: "middle",
    };
    addText(slide, label, { left: left - 30, top: 300, width: 140, height: 70 }, {
      fontSize: 21, bold: true, color: C.ink, align: "center",
    });
    return circle;
  });
  for (let i = 0; i < nodes.length - 1; i += 1) {
    slide.shapes.connect(nodes[i], nodes[i + 1], {
      kind: "straight",
      fromSide: "right",
      toSide: "left",
      line: { style: "solid", fill: C.gray, width: 2 },
      tail: { type: "arrow", width: "sm", length: "sm" },
    });
  }
  addText(slide, "Explicit state", { left: 120, top: 450, width: 230, height: 40 }, {
    fontSize: 25, bold: true, color: C.navy,
  });
  addText(slide, "query, route, evidence, answer, warnings, feedback, retry count", {
    left: 345, top: 450, width: 820, height: 40,
  }, { fontSize: 23, color: C.gray });
  addText(slide, "Termination condition", { left: 120, top: 515, width: 260, height: 40 }, {
    fontSize: 25, bold: true, color: C.navy,
  });
  addText(slide, "one retry, then a low-confidence human-review response", {
    left: 385, top: 515, width: 760, height: 40,
  }, { fontSize: 23, color: C.gray });
  addNotes(slide, [
    "Explain why routing is lexical: the business distinction does not require another model call.",
    "The LLM drives semantic synthesis, while code owns validation and termination.",
    "Source: src/allianz_rag/graph.py.",
  ]);
}

// 8. Structured answer and guardrails
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Structured answers make failures observable", 8);
  addText(slide, "Response contract", { left: 82, top: 165, width: 420, height: 50 }, {
    fontSize: 31, bold: true, color: C.navy,
  });
  addBullets(slide, [
    "Plain-language answer",
    "Parties and case type",
    "Agreement responsibility",
    "Applicable framework and key facts",
    "Missing information",
    "Citations, confidence and limitations",
  ], { left: 82, top: 235, width: 485, height: 350 }, { fontSize: 23, spaceAfter: 10 });
  const boundary = slide.shapes.add({
    geometry: "rect",
    position: { left: 625, top: 155, width: 3, height: 430 },
    fill: C.line,
    line: { fill: "none", width: 0 },
  });
  boundary.name = "Contract boundary";
  addText(slide, "Runtime checks", { left: 690, top: 165, width: 420, height: 50 }, {
    fontSize: 31, bold: true, color: C.navy,
  });
  addBullets(slide, [
    "Retrieved passages are untrusted data",
    "Every citation must match source, page and chunk ID",
    "Unknown citations trigger one regeneration",
    "Weak retrieval returns an explicit fallback",
    "High-impact actions remain outside the graph",
  ], { left: 690, top: 235, width: 485, height: 325 }, { fontSize: 23, spaceAfter: 12 });
  addText(slide, "A fluent answer does not pass unless its evidence is valid", {
    left: 170, top: 620, width: 940, height: 40,
  }, { fontSize: 27, bold: true, color: C.red, align: "center" });
  addNotes(slide, [
    "Avoid claiming that citation validation proves factual correctness. It proves that cited identifiers were retrieved.",
    "The next production control is an expert-labelled faithfulness and correctness evaluation.",
    "Source: models.py, prompts.py and graph.py.",
  ]);
}

// 9. Demo sequence
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Live demo sequence", 9);
  const cases = [
    ["1", "Factual question", "Does alcohol alone exclude the agreements?", C.blue],
    ["2", "Accident case", "Rear-end collision at a red light", C.blue],
    ["3", "Controlled limitation", "Parked car and unknown vehicle", C.red],
  ];
  cases.forEach(([number, heading, detail, color], index) => {
    const top = 165 + index * 150;
    const badge = slide.shapes.add({
      geometry: "ellipse",
      position: { left: 95, top, width: 70, height: 70 },
      fill: color,
      line: { fill: "none", width: 0 },
    });
    badge.text = number;
    badge.text.style = {
      typeface: family, fontSize: 26, bold: true, color: C.white,
      autoFit: "none", horizontalAlignment: "center", verticalAlignment: "middle",
    };
    addText(slide, heading, { left: 200, top: top - 4, width: 340, height: 42 }, {
      fontSize: 28, bold: true, color: C.navy,
    });
    addText(slide, detail, { left: 200, top: top + 42, width: 880, height: 45 }, {
      fontSize: 23, color: C.gray,
    });
  });
  addText(slide, "For each case: show the conclusion, missing facts and cited PDF page", {
    left: 105, top: 620, width: 1070, height: 44,
  }, { fontSize: 25, bold: true, color: C.navy, align: "center" });
  addNotes(slide, [
    "Target eight minutes for the full demo.",
    "Open one citation in the PDF for the first two examples. In the third, praise the fallback behavior rather than forcing an answer.",
    "Source: docs/interview_runbook.md and evaluation/golden_dataset.jsonl.",
  ]);
}

// 10. Evaluation table
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Evaluation separates retrieval from generation", 10);
  const table = slide.tables.add({
    rows: 5,
    columns: 3,
    left: 90,
    top: 170,
    width: 1100,
    height: 340,
    columnWidths: [230, 350, 520],
    values: [
      ["Layer", "Metric", "Question answered"],
      ["Retrieval", "Recall at 6", "Did any relevant page appear?"],
      ["Retrieval", "Mean reciprocal rank", "How early did relevant evidence appear?"],
      ["Generation", "Citation validity", "Do citations match retrieved metadata?"],
      ["Generation", "Structured completeness", "Are required fields present?"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({
    fill: C.navy,
    textStyle: { typeface: family, fontSize: 20, bold: true, color: C.white },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 3 }).assign({
    textStyle: { typeface: family, fontSize: 19, color: C.ink },
    margins: { left: 12, right: 12, top: 8, bottom: 8 },
    anchor: "middle",
  });
  addText(slide, "Ten golden cases cover the interview hints and key manual topics", {
    left: 120, top: 555, width: 1040, height: 48,
  }, { fontSize: 28, bold: true, color: C.navy, align: "center" });
  addText(slide, "Next step: expert labels for correctness, faithfulness and usefulness", {
    left: 165, top: 612, width: 950, height: 36,
  }, { fontSize: 21, color: C.gray, align: "center" });
  addNotes(slide, [
    "State that the project currently validates logic with 11 passing unit tests and Ruff.",
    "Do not invent retrieval scores before the complete embedding stack and index have run.",
    "Source: evaluation.py, golden_dataset.jsonl and local test output on 29 September 2026.",
  ]);
}

// 11. Five-day plan
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, "Five day implementation plan", 11);
  const table = slide.tables.add({
    rows: 6,
    columns: 3,
    left: 78,
    top: 150,
    width: 1120,
    height: 420,
    columnWidths: [100, 430, 590],
    values: [
      ["Day", "Outcome", "Acceptance check"],
      ["1", "Requirements and architecture", "Pages extract with metadata; decisions recorded"],
      ["2", "Ingestion and retrieval", "Golden questions return relevant pages in top six"],
      ["3", "Agent graph and guardrails", "Citations validate; every route terminates"],
      ["4", "API, UI, Docker and evaluation", "Examples run through one shared service"],
      ["5", "Tests, deliverables and rehearsal", "Clean setup and timed presentation"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 3 }).assign({
    fill: C.navy,
    textStyle: { typeface: family, fontSize: 20, bold: true, color: C.white },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 5, columnCount: 3 }).assign({
    textStyle: { typeface: family, fontSize: 19, color: C.ink },
    margins: { left: 12, right: 12, top: 8, bottom: 8 },
    anchor: "middle",
  });
  addText(slide, "Scope rule", { left: 140, top: 610, width: 150, height: 34 }, {
    fontSize: 21, bold: true, color: C.red,
  });
  addText(slide, "Complete one auditable vertical slice before adding breadth", {
    left: 290, top: 604, width: 820, height: 42,
  }, { fontSize: 24, bold: true, color: C.navy });
  addNotes(slide, [
    "Use this slide to demonstrate project management and acceptance criteria, not only a list of tasks.",
    "Source: docs/delivery_plan.md.",
  ]);
}

// 12. Risks and roadmap
{
  const slide = presentation.slides.add();
  slide.background.fill = C.navy;
  addText(slide, "Risks and production roadmap", {
    left: 72, top: 48, width: 1030, height: 64,
  }, { fontSize: 40, bold: true, color: C.white });
  addText(slide, "Immediate risks", { left: 82, top: 165, width: 420, height: 48 }, {
    fontSize: 30, bold: true, color: C.cyan,
  });
  addBullets(slide, [
    "Damaged PDF text layer",
    "2004 source may be outdated",
    "Model and network dependency",
    "No expert-labelled answer set yet",
  ], { left: 82, top: 235, width: 480, height: 290 }, {
    fontSize: 23, color: C.white, spaceAfter: 13,
  });
  addText(slide, "Production priorities", { left: 680, top: 165, width: 430, height: 48 }, {
    fontSize: 30, bold: true, color: C.cyan,
  });
  addBullets(slide, [
    "Spanish OCR and document versioning",
    "Hybrid retrieval and reranking",
    "Expert correctness and faithfulness labels",
    "Access control, PII redaction and audit traces",
    "Human approval before downstream actions",
  ], { left: 680, top: 235, width: 500, height: 330 }, {
    fontSize: 23, color: C.white, spaceAfter: 12,
  });
  addText(slide, "Repository outcome", { left: 85, top: 600, width: 245, height: 36 }, {
    fontSize: 21, bold: true, color: C.red,
  });
  addText(slide, "A testable MVP with explicit limits, ready for interview rehearsal", {
    left: 330, top: 592, width: 820, height: 50,
  }, { fontSize: 25, bold: true, color: C.white });
  addNotes(slide, [
    "Close by distinguishing what is complete from what remains before production use.",
    "Invite questions on retrieval evaluation, agent termination or deployment trade-offs.",
    "Source: docs/technical_approach.md and delivery_plan.md.",
  ]);
}

const requirements = {
  explicitTotalSlideCount: 12,
  requiredNativeTableOwnerSlides: [10, 11],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [family] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "10",
    "--require-native-table-slide", "11",
  ],
  requiredNativeTableOwnerSlides: [10, 11],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "interview-presentation-v2.validation.json"),
});

const finalDeck = await PresentationFile.importPptx(await FileBlob.load(FINAL_PPTX));
const previewDir = path.join(TMP_DIR, "final-previews");
await fs.mkdir(previewDir, { recursive: true });
for (let index = 0; index < finalDeck.slides.items.length; index += 1) {
  const slide = finalDeck.slides.items[index];
  const preview = await finalDeck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(
    path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(JSON.stringify({ finalPath: FINAL_PPTX, font: family, result }, null, 2));
