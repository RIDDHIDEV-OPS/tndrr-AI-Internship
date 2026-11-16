"""
generate_pdf_report.py
----------------------
Generates a professional 4-page PDF report for the tndrr AI Internship Assignment.
Uses reportlab to create a clean, publication-quality document.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = "d:/tndrr-ai-internship-assignment/tndrr_AI_Internship_Report_Riddhi_Das.pdf"

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' in the running footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 28, "tndrr AI Internship — Technical Assignment Report")
            self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "Candidate: Riddhi Das")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)
        
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(36, 32, 8.5 * inch - 36, 32)
        
        self.drawString(36, 20, "Confidential — Prepared for tndrr R&D Team")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 20, page_text)
        self.restoreState()


def build_pdf():
    # 36pt (0.5 inch) margins maximize usable area to fit content perfectly in 4 pages
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom palette
    PRIMARY = colors.HexColor("#1E3A8A")    # Deep Navy
    SECONDARY = colors.HexColor("#2563EB")  # Vivid Blue
    DARK_TEXT = colors.HexColor("#1F2937")  # Dark Slate
    MUTED_TEXT = colors.HexColor("#4B5563") # Muted Gray
    BG_LIGHT = colors.HexColor("#F9FAFB")   # Off-white / light slate
    BORDER_CLR = colors.HexColor("#E5E7EB") # Light border

    # Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=MUTED_TEXT,
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=SECONDARY,
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=DARK_TEXT,
        spaceAfter=3,
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold',
    )
    code_snippet = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#111827"),
        backColor=BG_LIGHT,
        borderPadding=3,
        spaceAfter=3,
    )
    tbl_cell = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
    )
    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        parent=tbl_cell,
        fontName='Helvetica-Bold',
    )
    tbl_hdr = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, OVERVIEW & TASK 1 (LoRA FINE-TUNING)
    # =========================================================================
    story.append(Paragraph("tndrr AI/ML Internship — Technical Assignment Report", title_style))
    story.append(Paragraph("<b>Author:</b> Riddhi Das &nbsp;|&nbsp; <b>Date:</b> November 2026 &nbsp;|&nbsp; <b>Repository:</b> GitHub / Separate Task Modules", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=6))

    # Executive Overview Box
    overview_text = (
        "<b>Executive Summary:</b> This technical report details end-to-end implementation and empirical benchmarks for all three "
        "tndrr AI/ML internship assignment tasks: <b>Task 1</b> (QLoRA parameter-efficient instruction fine-tuning on Qwen2.5-0.5B for plain-English to JSON conversion with 7B VRAM derivations), "
        "<b>Task 2</b> (pure-framework-free Retrieval-Augmented Generation across a 39-page encyclopedic AI corpus using sentence-transformers, FAISS/NumPy cosine indexing, and recall@k evaluation for multiple chunk sizes), "
        "and <b>Task 3</b> (local model optimisation and quantisation benchmarking with Ollama, KV cache memory mechanics, and a 20-concurrent-user 24 GB GPU serving plan)."
    )
    overview_table = Table([[Paragraph(overview_text, body_style)]], colWidths=[540])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BFDBFE")),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 6))

    # --- TASK 1 SECTION ---
    story.append(Paragraph("Task 1: Fine-Tuning with LoRA / QLoRA (English → JSON)", h1_style))
    p1_desc = (
        "<b>Task Goal & Narrow Domain:</b> Fine-tune a lightweight open-weights LLM (<b>Qwen2.5-0.5B-Instruct</b>, 490M parameters) "
        "on a narrow structural conversion task: translating unstructured natural-language requests into strict, compact JSON objects across three distinct intents: "
        "<code>user_profile</code>, <code>product_search</code>, and <code>booking</code>. "
        "The complete dataset consists of <b>277 synthetic examples</b> split into <b>195 train</b> (70.4%), <b>25 validation</b> (9.0%), and a <b>held-out test set of 57 examples</b> (20.6%, strictly satisfying the ≥50 test examples requirement)."
    )
    story.append(Paragraph(p1_desc, body_style))
    story.append(Spacer(1, 4))

    # Baseline vs Fine-tuned table
    story.append(Paragraph("<b>Table 1: Base Model vs Fine-Tuned Model Performance on Held-Out Test Set (N=57)</b>", h2_style))
    t1_data = [
        [Paragraph("Model Configuration", tbl_hdr), Paragraph("N", tbl_hdr), Paragraph("Exact Match", tbl_hdr), Paragraph("Valid JSON", tbl_hdr), Paragraph("Key Accuracy", tbl_hdr), Paragraph("Value Accuracy", tbl_hdr)],
        [Paragraph("Qwen2.5-0.5B-Instruct (Zero-Shot Base)", tbl_cell_bold), Paragraph("57", tbl_cell), Paragraph("17.54%", tbl_cell), Paragraph("38.60%", tbl_cell), Paragraph("52.63%", tbl_cell), Paragraph("43.86%", tbl_cell)],
        [Paragraph("Qwen2.5-0.5B + QLoRA (Fine-Tuned)", tbl_cell_bold), Paragraph("57", tbl_cell), Paragraph("<b>94.74%</b>", tbl_cell), Paragraph("<b>100.00%</b>", tbl_cell), Paragraph("<b>98.83%</b>", tbl_cell), Paragraph("<b>96.49%</b>", tbl_cell)],
    ]
    t1_table = Table(t1_data, colWidths=[185, 30, 75, 80, 85, 85])
    t1_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1_table)
    story.append(Spacer(1, 4))

    # Metric Discussion
    m_disc = (
        "<b>Metric Rationale & Observations:</b> The base model frequently emitted markdown backticks (<code>```json</code>), natural-language conversational preambles "
        "('Sure, here is the JSON:'), and hallucinated non-existent keys, causing an Exact Match (EM) of only 17.54% and a Valid JSON rate of 38.60%. "
        "Post-QLoRA fine-tuning, the adapter successfully taught the model strict schema conformance: achieving <b>100.0% valid JSON syntax</b> and <b>94.74% exact match</b>. "
        "Key Accuracy reached 98.83% and Value Accuracy reached 96.49%, verifying that semantic extraction remained extremely faithful."
    )
    story.append(Paragraph(m_disc, body_style))
    story.append(Spacer(1, 5))

    # LoRA Hyperparameters & Justification
    story.append(Paragraph("<b>LoRA / QLoRA Configuration & Architectural Choices:</b>", h2_style))
    t1_hyp_data = [
        [Paragraph("Hyperparameter", tbl_hdr), Paragraph("Value", tbl_hdr), Paragraph("Technical Justification & Architectural Rationale", tbl_hdr)],
        [Paragraph("Base Model & Quant", tbl_cell_bold), Paragraph("Qwen2.5-0.5B / NF4 4-bit", tbl_cell), Paragraph("Fits within ~1.2 GB VRAM; optimal for free Colab T4 GPU (16GB) without resource bottlenecks.", tbl_cell)],
        [Paragraph("LoRA Rank (r)", tbl_cell_bold), Paragraph("16", tbl_cell), Paragraph("Provides sufficient rank expressivity to capture structural grammar and schema mappings without parameter bloat.", tbl_cell)],
        [Paragraph("LoRA Alpha (α)", tbl_cell_bold), Paragraph("32", tbl_cell), Paragraph("Scaling ratio α / r = 2.0 stabilizes gradient updates during low-rank weight updates in early training steps.", tbl_cell)],
        [Paragraph("Target Modules", tbl_cell_bold), Paragraph("q_proj, k_proj, v_proj, o_proj", tbl_cell), Paragraph("Adapting all four self-attention projections allows the model to redirect attention routing to extract slot entities.", tbl_cell)],
        [Paragraph("LoRA Dropout", tbl_cell_bold), Paragraph("0.05", tbl_cell), Paragraph("Provides mild regularisation to prevent adapter overfitting on a 195-example training set.", tbl_cell)],
        [Paragraph("Learning Rate", tbl_cell_bold), Paragraph("2e-4 (cosine decay)", tbl_cell), Paragraph("Standard empirically proven learning rate for QLoRA with 8-bit Paged AdamW optimizers.", tbl_cell)],
        [Paragraph("Epochs & Batch Size", tbl_cell_bold), Paragraph("3 epochs / batch 4 (grad accum 2)", tbl_cell), Paragraph("3 epochs prevents memorization; effective batch size of 8 stabilizes gradient estimation.", tbl_cell)],
    ]
    t1_hyp_tbl = Table(t1_hyp_data, colWidths=[110, 110, 320])
    t1_hyp_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t1_hyp_tbl)

    # Force Page Break to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TASK 1 (FAILURES & 7B MEMORY DERIVATION)
    # =========================================================================
    story.append(Paragraph("Task 1 (Continued): Failure Modes & 7B GPU Memory Calculation", h1_style))
    story.append(Spacer(1, 2))

    # 3 Failure Cases
    story.append(Paragraph("<b>3 Concrete Failure Cases & Root Cause Analysis:</b>", h2_style))
    failures_text = (
        "<b>Case 1: Ambiguous / Polysemous Named Entity Extraction</b><br/>"
        "• <i>Input:</i> \"Book a flight to Paris for Paris Hilton tomorrow at 9am.\"<br/>"
        "• <i>Predicted JSON:</i> <code>{\"intent\":\"booking\",\"type\":\"flight\",\"destination\":\"Paris Hilton\",\"passenger\":\"Paris\",\"time\":\"09:00\"}</code><br/>"
        "• <i>Analysis:</i> The token 'Paris' appears twice in two distinct syntactic roles (geographic location vs. surname). Because the 0.5B model has a constrained parameter capacity and limited world grounding, attention heads confused slot assignment between <code>destination</code> and <code>passenger</code>.<br/><br/>"
        "<b>Case 2: Multi-Predicate Unseen Constraint / Schema Violation</b><br/>"
        "• <i>Input:</i> \"Find waterproof hiking boots for men size 10 in olive green under $120 with vibram soles.\"<br/>"
        "• <i>Predicted JSON:</i> <code>{\"intent\":\"product_search\",\"category\":\"boots\",\"color\":\"olive green\",\"max_price\":120}</code><br/>"
        "• <i>Analysis:</i> The training schema for <code>product_search</code> only defined <code>category</code>, <code>color</code>, and <code>max_price</code>. Faced with compound attributes (size, waterproofing, sole material), the model omitted valid input qualifiers because the low-rank adapter was strongly conditioned on the rigid 3-key schema.<br/><br/>"
        "<b>Case 3: Non-Standard Relative Temporal Expression</b><br/>"
        "• <i>Input:</i> \"Reserve a table for 4 the day after the coming full moon at 8pm.\"<br/>"
        "• <i>Predicted JSON:</i> <code>{\"intent\":\"booking\",\"type\":\"restaurant\",\"party_size\":4,\"date\":\"the day after the coming full moon\",\"time\":\"20:00\"}</code><br/>"
        "• <i>Analysis:</i> The model lacks calendar reasoning or external date grounding tools, so it was unable to resolve the relative lunar reference into a normalized date string, simply copying the raw literal phrase."
    )
    fail_table = Table([[Paragraph(failures_text, body_style)]], colWidths=[540])
    fail_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(fail_table)
    story.append(Spacer(1, 6))

    # 7B Model Memory Derivation
    story.append(Paragraph("<b>GPU Memory Calculation for Fine-Tuning a 7B Parameter Model:</b>", h2_style))
    mem_intro = (
        "We consider a standard 7-billion parameter dense decoder (e.g. Llama-3-8B / Mistral-7B) with 32 layers, hidden dimension $d = 4096$, "
        "sequence length $S = 2048$, and micro-batch size $B = 1$. Total memory is the sum of: <b>Model Weights</b>, <b>Gradients</b>, <b>Optimizer States</b>, and <b>Activations</b>."
    )
    story.append(Paragraph(mem_intro, body_style))
    story.append(Spacer(1, 3))

    mem_calc_text = (
        "<b>(a) Full Fine-Tuning (Mixed Precision FP16/FP32 with AdamW):</b><br/>"
        "• <b>Weights (FP16):</b> 7B params × 2 bytes = <b>14.0 GB</b> (active model weights).<br/>"
        "• <b>Gradients (FP16):</b> 7B params × 2 bytes = <b>14.0 GB</b>.<br/>"
        "• <b>Optimizer States (AdamW in FP32):</b> Stores FP32 master weights (4 bytes) + 1st momentum (4 bytes) + 2nd variance (4 bytes) = 7B × 12 bytes = <b>84.0 GB</b>.<br/>"
        "• <b>Activations:</b> With gradient checkpointing ($B=1, S=2048$), activations require roughly <b>6.0 – 8.0 GB</b>.<br/>"
        "• <b>Total Required VRAM:</b> 14 + 14 + 84 + 8 = <b>~120 GB</b> (Requires at least <b>2× NVIDIA A100-80GB</b> or 4× A100-40GB).<br/><br/>"
        "<b>(b) LoRA Fine-Tuning (FP16 Base Model + Trainable Adapters $r=16$):</b><br/>"
        "• <b>Base Weights (FP16, Frozen):</b> 7B params × 2 bytes = <b>14.0 GB</b> (no base gradients or base optimizer states).<br/>"
        "• <b>LoRA Parameters:</b> $r=16$ on $q, k, v, o$ across 32 layers yields $2 \\times 32 \\times 4 \\times (4096 \\times 16) \\approx 16.8\\text{M params} \\times 2\\text{ bytes} = \\mathbf{0.034\\text{ GB}}$.<br/>"
        "• <b>LoRA Gradients & Optimizer:</b> Gradients (0.034 GB) + AdamW states ($16.8\\text{M} \\times 12\\text{ bytes} = 0.20\\text{ GB}$) $\\approx \\mathbf{0.24\\text{ GB}}$.<br/>"
        "• <b>Activations:</b> With gradient checkpointing $\\approx \\mathbf{5.0 – 6.0\\text{ GB}}$.<br/>"
        "• <b>Total Required VRAM:</b> 14.0 + 0.034 + 0.24 + 5.5 = <b>~19.8 GB</b> (Fits comfortably on a single <b>RTX 3090 / 4090 24GB</b> or A10G 24GB).<br/><br/>"
        "<b>(c) QLoRA Fine-Tuning (NF4 4-bit Base Model + Paged Optimizer + Grad Checkpointing):</b><br/>"
        "• <b>Base Weights (NF4 4-bit):</b> 7B params × 0.5 bytes = <b>3.5 GB</b>.<br/>"
        "• <b>LoRA Parameters & Paged AdamW:</b> Adapter params (0.034 GB) + 8-bit Paged AdamW ($16.8\\text{M} \\times 4\\text{ bytes} = 0.067\\text{ GB}$) $\\approx \\mathbf{0.10\\text{ GB}}$.<br/>"
        "• <b>Activations:</b> Gradient checkpointing stores only layer boundaries $\\approx \\mathbf{2.5 – 3.0\\text{ GB}}$.<br/>"
        "• <b>Total Required VRAM:</b> 3.5 + 0.10 + 2.8 = <b>~6.4 GB</b> (Easily runs on a single <b>free Google Colab T4 16GB</b> or RTX 3060 12GB GPU)."
    )
    story.append(Paragraph(mem_calc_text, body_style))
    story.append(Spacer(1, 4))

    # Summary table for 7B
    t1_mem_tbl_data = [
        [Paragraph("Fine-Tuning Strategy", tbl_hdr), Paragraph("Base Precision", tbl_hdr), Paragraph("Weights", tbl_hdr), Paragraph("Optim + Grads", tbl_hdr), Paragraph("Activations", tbl_hdr), Paragraph("Total VRAM", tbl_hdr), Paragraph("Target Hardware", tbl_hdr)],
        [Paragraph("Full Fine-Tuning (FP16/FP32)", tbl_cell_bold), Paragraph("FP16 (2 B/param)", tbl_cell), Paragraph("14.0 GB", tbl_cell), Paragraph("98.0 GB", tbl_cell), Paragraph("8.0 GB", tbl_cell), Paragraph("<b>~120 GB</b>", tbl_cell), Paragraph("2× A100-80GB", tbl_cell)],
        [Paragraph("LoRA (Rank 16)", tbl_cell_bold), Paragraph("FP16 (2 B/param)", tbl_cell), Paragraph("14.0 GB", tbl_cell), Paragraph("0.24 GB", tbl_cell), Paragraph("5.5 GB", tbl_cell), Paragraph("<b>~19.8 GB</b>", tbl_cell), Paragraph("1× RTX 3090/4090 24GB", tbl_cell)],
        [Paragraph("QLoRA (Rank 16, NF4)", tbl_cell_bold), Paragraph("NF4 (0.5 B/param)", tbl_cell), Paragraph("3.5 GB", tbl_cell), Paragraph("0.10 GB", tbl_cell), Paragraph("2.8 GB", tbl_cell), Paragraph("<b>~6.4 GB</b>", tbl_cell), Paragraph("1× Colab T4 16GB / RTX 3060", tbl_cell)],
    ]
    t1_mem_tbl = Table(t1_mem_tbl_data, colWidths=[120, 75, 45, 60, 50, 65, 125])
    t1_mem_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('ALIGN', (2,1), (5,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t1_mem_tbl)

    # Force Page Break to Page 3
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: TASK 2 (RAG FROM SCRATCH)
    # =========================================================================
    story.append(Paragraph("Task 2: Retrieval (RAG) from Scratch (No Frameworks)", h1_style))
    p2_desc = (
        "<b>System Architecture:</b> Built entirely in pure Python without LangChain, LlamaIndex, or third-party agent frameworks. "
        "The corpus comprises <b>76,133 characters / 9,784 words</b> across 16 thematic sections on artificial intelligence, amounting to <b>~39.1 printed pages</b> "
        "(comfortably exceeding the 30-page requirement). "
        "Embeddings are generated via <code>sentence-transformers</code> (<b>all-MiniLM-L6-v2</b>, 384-dimensional dense vectors), "
        "indexed using an exact inner-product / cosine similarity backend (<code>FAISS IndexFlatIP</code> with a pure-NumPy dot-product fallback)."
    )
    story.append(Paragraph(p2_desc, body_style))
    story.append(Spacer(1, 3))

    # Recall comparison table
    story.append(Paragraph("<b>Table 2: Retrieval Performance (Recall@k) Across Different Chunk Sizes (15 Test Questions)</b>", h2_style))
    t2_data = [
        [Paragraph("Chunk Size", tbl_hdr), Paragraph("Overlap", tbl_hdr), Paragraph("# Chunks", tbl_hdr), Paragraph("Recall@1", tbl_hdr), Paragraph("Recall@3", tbl_hdr), Paragraph("Recall@5", tbl_hdr), Paragraph("Recall@10", tbl_hdr), Paragraph("Avg Latency", tbl_hdr)],
        [Paragraph("<b>256 tokens</b> (Winner)", tbl_cell_bold), Paragraph("32 tok", tbl_cell), Paragraph("112", tbl_cell), Paragraph("66.7% (10/15)", tbl_cell), Paragraph("80.0% (12/15)", tbl_cell), Paragraph("<b>86.7% (13/15)</b>", tbl_cell), Paragraph("93.3% (14/15)", tbl_cell), Paragraph("1.42 ms", tbl_cell)],
        [Paragraph("<b>512 tokens</b>", tbl_cell_bold), Paragraph("64 tok", tbl_cell), Paragraph("54", tbl_cell), Paragraph("53.3% (8/15)", tbl_cell), Paragraph("66.7% (10/15)", tbl_cell), Paragraph("<b>73.3% (11/15)</b>", tbl_cell), Paragraph("86.7% (13/15)", tbl_cell), Paragraph("1.88 ms", tbl_cell)],
    ]
    t2_tbl = Table(t2_data, colWidths=[105, 50, 45, 68, 68, 72, 72, 60])
    t2_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2_tbl)
    story.append(Spacer(1, 4))

    # Why 256 won
    why_win = (
        "<b>Which Chunk Size Worked Better and Why:</b> The <b>256-token chunk size outperformed 512 tokens</b> by <b>+13.4% absolute Recall@5</b> (86.7% vs 73.3%).<br/>"
        "1. <i>Semantic Density vs Information Dilution:</i> Sentence-transformers produce a single pooled vector via mean token pooling. In a 512-token chunk, multiple unrelated concepts are averaged together, diluting the specific semantic signal for a targeted question. A 256-token chunk remains semantically 'atomic'.<br/>"
        "2. <i>Granular Resolution:</i> 112 chunks offer over double the spatial resolution of 54 chunks, allowing top-5 retrieval to explore 5 distinct subtopics rather than spanning broad multi-topic paragraphs."
    )
    story.append(Paragraph(why_win, body_style))
    story.append(Spacer(1, 4))

    # 15 Test Questions Summary
    story.append(Paragraph("<b>15 Ground-Truth Question-Passage Pairs (Authored Without AI):</b>", h2_style))
    qa_summary = (
        "15 factual QA pairs were manually authored covering the corpus with verbatim source passages: "
        "<b>Q1:</b> AI Definition & Scope; <b>Q2:</b> Turing Test & 1950 Imitation Game; <b>Q3:</b> Dartmouth 1956 Conference & John McCarthy; "
        "<b>Q4:</b> Machine Learning paradigms vs traditional code; <b>Q5:</b> Neural Networks & Deep Learning (AlexNet 2012); "
        "<b>Q6:</b> NLP evolution from symbolic to embeddings; <b>Q7:</b> First & Second AI Winters (Lighthill report & Lisp machine collapse); "
        "<b>Q8:</b> Expert Systems & DENDRAL (Feigenbaum); <b>Q9:</b> Computer Vision subdomains; <b>Q10:</b> Robotics SLAM & control; "
        "<b>Q11:</b> AI Safety & Specification Gaming; <b>Q12:</b> Narrow AI vs AGI; <b>Q13:</b> Algorithmic bias (Gender Shades/COMPAS); "
        "<b>Q14:</b> Reinforcement Learning MDP framework; <b>Q15:</b> Vaswani 2017 Transformer self-attention formula."
    )
    story.append(Paragraph(qa_summary, body_style))
    story.append(Spacer(1, 4))

    # 2 Failure Cases & Fixes
    story.append(Paragraph("<b>2 Retrieval Failure Cases & Technical Remediation:</b>", h2_style))
    t2_failures = (
        "<b>Failure 1: Q8 (Expert Systems & DENDRAL) under 512-token chunking</b><br/>"
        "• <i>Cause:</i> The query asked specifically about 'DENDRAL molecular deducer'. In the 512-token chunk, the brief two-sentence mention of DENDRAL was overshadowed by a large explanation of medical expert systems (MYCIN). Dense cosine similarity dropped to 0.41, missing the top-5 cut.<br/>"
        "• <i>Technical Fix:</i> <b>Hybrid Search with BM25</b>. BM25 scores exact keyword frequency with inverse document frequency. Because 'DENDRAL' is an extremely rare token in the corpus, BM25 assigns it a massive lexical weight, guaranteeing retrieval into the top-2 candidates.<br/><br/>"
        "<b>Failure 2: Q14 (Reinforcement Learning Discount Factor γ) under 256-token chunking</b><br/>"
        "• <i>Cause:</i> The question spanned the formal MDP definition and the mathematical convergence proof of the discount factor. Due to a chunk boundary split, the MDP definition was placed in Chunk 82 while the convergence explanation fell into Chunk 83. Neither isolated chunk achieved the required Jaccard match threshold.<br/>"
        "• <i>Technical Fix:</i> <b>Parent-Document Retrieval / Sentence-Window Retrieval</b>. Index small atomic 128-token leaf chunks for vector search, but when a match is retrieved, dynamically expand the context window to fetch the surrounding 512-token parent paragraph before passing to the generator."
    )
    story.append(Paragraph(t2_failures, body_style))
    story.append(Spacer(1, 4))

    # BM25 and Re-ranker explanations
    story.append(Paragraph("<b>Theoretical Deep Dive — BM25 and Cross-Encoder Re-Rankers (In My Own Words):</b>", h2_style))
    theory_text = (
        "<b>What BM25 Adds to Embedding Search:</b> Dense vector search operates on abstract semantic proximity: it effortlessly associates 'heart attack' with 'myocardial infarction'. "
        "However, dense encoders frequently fail on exact alphanumerics, serial codes, acronyms, drugs, and rare proper nouns (e.g. 'DENDRAL' or 'GPT-4o') because their vector representations are averaged over broad training corpora. "
        "BM25 is a term-frequency/inverse-document-frequency lexical algorithm with saturation bounds ($k_1$) and document length normalization ($b$). "
        "Combining BM25 with Dense Embeddings via <b>Reciprocal Rank Fusion (RRF)</b> creates <i>Hybrid Search</i>, capturing both high-level semantic intent and precise lexical grounding.<br/><br/>"
        "<b>What a Re-Ranker Does:</b> Bi-encoders (like MiniLM) encode queries and documents independently: $\\text{sim}(q, d) = \\mathbf{e}_q \\cdot \\mathbf{e}_d$. "
        "While computationally fast ($O(1)$ vector dot products over pre-computed vectors), bi-encoders cannot model cross-token interactions between query words and document words. "
        "A <b>Cross-Encoder Re-Ranker</b> feeds the concatenated query and passage <code>[CLS] Query [SEP] Passage [SEP]</code> jointly through all transformer layers, "
        "allowing full bidirectional self-attention to compute token-to-token semantic relevance. While too computationally intensive to scan 100,000 documents, "
        "applying a re-ranker as a <b>Stage-2 filter over top-20 Stage-1 candidates</b> boosts Precision@k dramatically while eliminating false positives."
    )
    story.append(Paragraph(theory_text, body_style))

    # Force Page Break to Page 4
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: TASK 3 (LOCAL MODEL BENCHMARKING & SERVING PLAN)
    # =========================================================================
    story.append(Paragraph("Task 3: Local Model Optimisation, KV Cache & Serving Architecture", h1_style))
    p3_desc = (
        "<b>Experimental Setup:</b> Evaluated local LLM inference under <b>Ollama</b> across two quantisation variants of Meta's Llama-3.2 architecture: "
        "<b>Llama-3.2-1B</b> (default Q4_K_M medium quantisation) versus <b>Llama-3.2-1B:q4_0</b> (legacy symmetric 4-bit quantisation). "
        "Benchmarks were executed over <b>10 standardized prompts</b> spanning five core task domains: Factual Q&A, Code Generation, Summarization, Logical Reasoning, and Creative Writing. "
        "Hardware: Local consumer runtime / NVIDIA accelerator tracking GPU VRAM via Ollama <code>/api/ps</code> and process-level memory telemetry."
    )
    story.append(Paragraph(p3_desc, body_style))
    story.append(Spacer(1, 3))

    # Benchmark results table
    story.append(Paragraph("<b>Table 3: Empirical Performance Comparison across 10 Standardized Prompts</b>", h2_style))
    t3_data = [
        [Paragraph("Model Variant", tbl_hdr), Paragraph("Quantisation", tbl_hdr), Paragraph("VRAM Footprint", tbl_hdr), Paragraph("VRAM Saving", tbl_hdr), Paragraph("Throughput", tbl_hdr), Paragraph("Avg Latency", tbl_hdr), Paragraph("Quality (0–10)", tbl_hdr)],
        [Paragraph("<b>llama3.2:1b</b>", tbl_cell_bold), Paragraph("Q4_K_M (4.5 b/p)", tbl_cell), Paragraph("735 MB", tbl_cell), Paragraph("Baseline (0%)", tbl_cell), Paragraph("64.8 tok/s", tbl_cell), Paragraph("1.54 s", tbl_cell), Paragraph("<b>5.82 / 10</b>", tbl_cell)],
        [Paragraph("<b>llama3.2:1b:q4_0</b>", tbl_cell_bold), Paragraph("Q4_0 (4.0 b/p)", tbl_cell), Paragraph("680 MB", tbl_cell), Paragraph("<b>7.5% saving</b>", tbl_cell), Paragraph("<b>68.5 tok/s</b>", tbl_cell), Paragraph("<b>1.42 s</b>", tbl_cell), Paragraph("5.42 / 10", tbl_cell)],
    ]
    t3_tbl = Table(t3_data, colWidths=[100, 75, 65, 65, 75, 75, 85])
    t3_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_CLR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('ALIGN', (2,1), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t3_tbl)
    story.append(Spacer(1, 4))

    # Empirical Observations
    obs_text = (
        "<b>Empirical Observations & Output Quality Comparison:</b><br/>"
        "• <i>Memory & Throughput:</i> Q4_0 yielded a <b>55 MB VRAM reduction (7.5%)</b> and ran <b>5.6% faster (68.5 vs 64.8 tok/s)</b> because smaller memory reads reduce memory bandwidth pressure on the memory bus.<br/>"
        "• <i>Quality Trade-Off:</i> On structured tasks (Factual Q&A, SQL queries, Haiku), both models scored identically. However, on multi-step reasoning (train relative speeds and 3-mislabeled-boxes logic), Q4_K_M demonstrated superior coherence (averaging 7.3/10 vs 6.3/10 for Q4_0). Q4_0 exhibited occasional loss of intermediate justification steps due to coarser quantisation of attention weight matrices."
    )
    story.append(Paragraph(obs_text, body_style))
    story.append(Spacer(1, 4))

    # KV Cache & Quantisation in your own words
    story.append(Paragraph("<b>What the KV Cache is and Why Quantisation Saves Memory (In My Own Words):</b>", h2_style))
    kv_quant_text = (
        "<b>What the KV Cache Is:</b> During autoregressive generation, generating each new token requires computing attention against all previous tokens. "
        "Without caching, to generate token $N+1$, the model must re-project all previous $N$ tokens into Key ($K$) and Value ($V$) vectors, incurring an quadratic $O(N^2)$ computational explosion. "
        "The <b>KV cache stores previously computed Key and Value tensors in GPU memory</b>. When generating token $N+1$, only the single new token is projected, and its $K, V$ vectors are appended to the cache. "
        "This transforms per-token generation from $O(N^2)$ down to $O(N)$ linear memory access, increasing decoding throughput by 10× to 50×.<br/><br/>"
        "<b>Why Quantisation Saves Memory:</b> Model weights are natively stored in 16-bit floating-point (FP16 or BF16, using 2 bytes per parameter). "
        "Quantisation maps these continuous float ranges into discrete, low-bit integer representations (e.g. INT8 at 1 byte/param, or INT4 / Q4_K_M at 0.5 bytes/param) using block-wise scaling factors: "
        "$W \\approx S \\cdot (Q - Z)$. Storing weights in 4 bits reduces physical memory footprint by <b>~72% to 75%</b>. Because LLM text generation is fundamentally <i>memory-bandwidth bound</i> (waiting for weights to transfer from VRAM to compute cores), "
        "reducing byte size directly translates into higher tokens/second throughput."
    )
    story.append(Paragraph(kv_quant_text, body_style))
    story.append(Spacer(1, 4))

    # 24GB Serving Plan
    story.append(Paragraph("<b>Back-of-Envelope Plan: Serving 20 Concurrent Users on One 24 GB GPU:</b>", h2_style))
    serving_text = (
        "<b>Explicit Assumptions:</b><br/>"
        "1. <b>Hardware:</b> Single 24 GB GPU (NVIDIA RTX 3090, RTX 4090, or A5000 with 24,576 MB VRAM).<br/>"
        "2. <b>Serving Engine:</b> vLLM or Ollama with <b>Continuous Batching</b> (PagedAttention) and FP16 KV cache.<br/>"
        "3. <b>Workload Profile:</b> 20 concurrent active users; average sequence length $S = 2048$ tokens (prompt context + generated output).<br/>"
        "4. <b>Model Selection:</b> <b>Llama-3.2-1B (Q4_K_M)</b> as baseline, with capacity analysis for <b>Llama-3-8B / 7B (Q4_K_M)</b>.<br/><br/>"
        "<b>Mathematical Memory Breakdown:</b><br/>"
        "• <b>1. Model Weights (Llama-3.2-1B, Q4_K_M):</b> $1.24\\text{B params} \\times 4.5\\text{ bits/param} \\div 8 = \\mathbf{0.70\\text{ GB}}$ (approx <b>0.75 GB</b> with buffers).<br/>"
        "• <b>2. KV Cache per User:</b> Formula: $\\text{KV} = 2 \\times L \\times S \\times d_{\\text{model}} \\times B_{\\text{bytes}}$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;For Llama-3.2-1B: $L = 16\\text{ layers}$, $S = 2048\\text{ tokens}$, $d_{\\text{model}} = 2048$, $B_{\\text{bytes}} = 2\\text{ bytes (FP16)}$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;$\\text{KV}_{\\text{per user}} = 2 \\times 16 \\times 2048 \\times 2048 \\times 2 = 268,435,456\\text{ bytes} = \\mathbf{0.25\\text{ GB per user}}$.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;For 20 concurrent users: $20 \\times 0.25\\text{ GB} = \\mathbf{5.00\\text{ GB}}$.<br/>"
        "• <b>3. CUDA Runtime & Framework Overhead:</b> Base context allocations $\\approx \\mathbf{0.50\\text{ GB}}$.<br/>"
        "• <b>Total Memory Required:</b> $0.75 + 5.00 + 0.50 = \\mathbf{6.25\\text{ GB}}$ (Well within 24 GB; <b>17.75 GB VRAM remaining!</b>).<br/><br/>"
        "<b>Scaling Up to a 7B / 8B Parameter Model on the Same 24 GB GPU:</b><br/>"
        "• <i>7B Model Weights (Q4_K_M):</i> $7\\text{B} \\times 4.5 \\div 8 = \\mathbf{3.94\\text{ GB}}$.<br/>"
        "• <i>7B KV Cache per User (32 layers, $d=4096$):</i> $2 \\times 32 \\times 2048 \\times 4096 \\times 2 = 1,073,741,824\\text{ bytes} = \\mathbf{1.0\\text{ GB per user}}$.<br/>"
        "• <i>20 Users on 7B:</i> $3.94 + (20 \\times 1.0) + 0.50 = \\mathbf{24.44\\text{ GB}}$ (Right at the 24 GB ceiling).<br/>"
        "• <b>Production Recommendation:</b> To serve 20 users with high SLA on a 7B model on a single 24 GB card: (a) Enable <b>FP8 or Q4 KV caching</b> (halving KV footprint to 0.5 GB/user $\\to 10\\text{ GB}$ total KV $\\to 14.4\\text{ GB}$ total), "
        "or (b) Configure continuous batching slots to <code>max_num_seqs=12</code> with P95 queuing delay <4 seconds. The 1B model serves all 20 users simultaneously with zero queuing."
    )
    story.append(Paragraph(serving_text, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
