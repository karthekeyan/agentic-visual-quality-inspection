# Agentic Visual Quality Inspection for Casting Manufacturing

## Functional Proposal

### 1. Overview

This proposal describes an **AI-based visual quality inspection system for cast metal parts**.

Instead of simply saying **"defective" or "not defective,"** the system:

* Detects the defect
* Describes the pattern of the defect
* Suggests possible root causes, grounded in a curated knowledge base
* Recommends what to do with the part, including escalation to a human when appropriate
* Keeps a record for traceability

The system uses a **trained vision model to detect defects** and **AI agents to describe, reason about, and act on** what it sees. This proof of concept has been built and validated end-to-end, including a working browser-based inspection console.

---

## 2. Problem Statement

Today, visual inspection in many casting plants is done manually. This creates some problems:

* **Inconsistent inspection**
  Results may vary between inspectors, shifts, or due to fatigue.

* **Poor traceability**
  Parts may be accepted or rejected without properly recording what defect was found and why it happened.

* **No proper feedback to the process**
  Defect information is often not used effectively to improve casting parameters such as:

  * Mold temperature
  * Pouring speed
  * Cooling rate

**Result:** Defects are detected, but the manufacturing process does not learn enough to prevent them in the future.

---

## 3. Proposed Solution

The proposed solution is an **agentic AI visual inspection system**.

The system separates two main activities:

### Vision Model → **Sees**

* Detects defects in the image
* Identifies where the defect is located (Grad-CAM heatmap)

### AI Agents → **Think and decide**

* Describe the defect pattern
* Find possible root causes
* Recommend what to do with the part
* Monitor defect trends
* Create inspection records

This separation makes the system easier to **improve, monitor, and maintain** — and keeps clear which parts of the system are trained machine learning, which are fixed rules, and which are AI-generated reasoning.

---

## 4. Agent Roles and How Each One Actually Works

| Agent | Function | Mechanism |
| --- | --- | --- |
| **Inspection Agent** | Runs the vision model on the image; returns a confidence score and defect heatmap. | Trained CNN (ResNet18) inference. |
| **Characterization Agent** | Describes the defect pattern (spread, position) in plain language. | Rule-based logic on heatmap geometry — no learning, no LLM. |
| **Root-Cause Agent** | Proposes a likely process-related cause. | Retrieval (ChromaDB) + LLM reasoning (Claude), grounded in a curated knowledge base. |
| **Disposition Agent** | Recommends accept / rework / scrap, or escalates uncertain cases. | Rule-based thresholds on confidence and defect pattern. |
| **Trend Agent** | Tracks defect patterns across recent inspections. | Rolling statistics over inspection history. |
| **Reporting Agent** | Compiles a structured, auditable inspection record. | Structured formatting of all agent outputs. |
| **Orchestrator** | Runs the agents in sequence, handles errors, flags cases needing human review. | LangGraph pipeline with error handling. |

**Only one agent — Root-Cause — uses a language model to generate its output.** The others use the trained vision model directly, or fixed, inspectable rules. This distinction matters for understanding what the system has learned versus what it has been explicitly programmed to do.

---

## 5. How the System Works

### Step 1: Image Inspection

A camera or file upload provides the part image. The **Inspection Agent** runs the trained model and returns a label (ok/defective), a confidence score, and a heatmap showing which region of the image most influenced the decision.

### Step 2: Defect Characterization

The **Characterization Agent** describes the heatmap's pattern — whether the highlighted region is small and concentrated or broad and diffuse, and whether it sits centered on the part or near an edge. This runs only for defective predictions and is explicitly flagged as a **heuristic approximation**, since the underlying dataset has no ground-truth defect-subtype labels (see Section 8).

### Step 3: Root Cause Analysis

The **Root-Cause Agent** retrieves the most relevant entries from a curated knowledge base (see Section 8) and asks an LLM to explain the likely process cause, grounded strictly in that retrieved content. The agent has been observed to weigh multiple candidate causes, favor the best-supported one, and explicitly recommend human verification rather than asserting false certainty.

### Step 4: Decide What to Do

The **Disposition Agent** recommends **accept**, **rework**, or **scrap**, based on confidence and defect pattern, with the reasoning behind each decision recorded for audit. If confidence is below a defined threshold, the case is flagged **`human_review_required`** rather than automatically resolved.

### Step 5: Monitor Defect Trends

The **Trend Agent** tracks defect and scrap rates across recent inspections and flags when the rate exceeds a configurable threshold.

### Step 6: Create an Inspection Record

The **Reporting Agent** compiles a structured record — defect description, root cause explanation, disposition, and supporting image — saved as a persistent, timestamped file for traceability.

### Step 7: Coordinate Everything

The **Orchestrator** runs all agents in order, catches failures in any individual agent so the pipeline does not crash entirely, and ensures escalation cases are clearly flagged rather than silently resolved.

---

## 6. Validated Results

### Detection Model (ResNet18)

* **Test accuracy: 99.86%**, validated on a held-out test set with confirmed zero data leakage between train/validation/test splits.
* **Precision: 100%**, **Recall: 99.78%**, **F1: 99.89%**, **ROC-AUC: 1.0**.
* **1 false negative** out of 715 test images — the model's only miss, at 63% confidence (a genuinely uncertain call, not a confidently wrong one, confirmed via Grad-CAM analysis).

### Explainability (Grad-CAM)

* Verified that correctly-classified defective images consistently activate the bore/inner-rim region of the part.
* The one false-negative case showed heatmap activation *away* from that trusted region — consistent with genuine model uncertainty.
* One flag from this review: a low-confidence correct "ok" call showed heatmap activation on background, not the part — a caution for trust calibration on genuinely novel images.

### GPU Inference Performance

* Benchmarked on an NVIDIA Titan RTX: **~2ms per image (single-image mean latency)**, **~500 images/second**, scaling to **~2,864 images/second** at batch size 32.
* Two independent measurement methods (single-image and batch-derived) agreed within measurement noise.
* **This reflects a desktop/workstation GPU.** It establishes technical feasibility and a foundation for future edge deployment, but does not predict performance on edge hardware (e.g. Jetson-class devices) without separate benchmarking there.

### Architecture Comparison

* ResNet18 and MobileNetV2 were trained and evaluated identically. Classification quality was equivalent (both missed the same single test case).
* ResNet18 was faster on this GPU (2.04ms vs. 3.72ms) — a counter-intuitive but real finding, since MobileNetV2's efficiency advantage depends on constrained hardware, not a discrete GPU.
* MobileNetV2 remains the stronger candidate specifically for future edge deployment (5x smaller, 5x fewer parameters), pending real edge-hardware benchmarking.
* **Decision: ResNet18 is the current production model.**

### Multi-Agent Pipeline

* All six agents chained and verified end-to-end, including on a 20-image batch run with zero pipeline errors.
* The Disposition Agent's escalation threshold was confirmed to correctly flag the model's known false-negative case for human review — a concrete demonstration that the agent layer adds a real safety check beyond the ML layer alone.
* **66 automated tests passing** across the data pipeline, model training, inference, and all six agents.

### Dashboard

* A browser-based inspection console (FastAPI backend, React frontend) was built and visually verified end-to-end: image upload, Grad-CAM overlay with defect localization, sequential readout of each agent's output, and clear escalation flagging for human review.

---

## 7. Expected Outcomes

The system can provide:

* **More consistent inspection** across different shifts and inspectors.
* **A demonstrated safety net**: the agent layer has been shown to catch at least one case the ML model alone got wrong.
* **Better traceability** of defects and inspection decisions, with a persistent audit record per inspection.
* **A foundation for connecting defect history to manufacturing process variables**, once real process data is available.
* **Human involvement in uncertain cases**, rather than full automation of every decision.

---

## 8. Honest Scope and Limitations

This section is deliberately explicit, since several parts of the system are architected but not yet validated against real-world ground truth.

### What is genuinely validated

* Binary defect detection (ok/defective): real, tested, high-accuracy.
* GPU inference speed: real, measured, reproducible.
* The Disposition Agent's escalation logic: demonstrated to work correctly on a known case.

### What is heuristic, not learned

* **Defect characterization** (e.g., "distributed irregularity") describes the *shape and position* of the model's attention pattern, not a validated defect-subtype classification. The dataset used (a public Kaggle set of casting images) contains only binary ok/defective labels — no porosity/crack/misrun ground truth exists to train or validate a true subtype classifier. A defect-subtype-labeled dataset would be required to close this gap.

### What is grounded but not independently verified

* **Root-cause reasoning** is grounded in a small, hand-curated knowledge base (9 entries, drawn from general casting/metallurgy knowledge, not learned from this dataset) via retrieval-augmented generation. The reasoning has been observed to be internally consistent, appropriately hedged, and correctly limited to the retrieved context — but its real-world correctness has not been validated against domain-expert review or real process records, because neither is available in this project. The system consistently recommends human verification rather than asserting unconfirmed causes as fact.

### What is simulated

* **Trend and batch data**: the dataset has no real batch, line, or timestamp metadata. Trend tracking uses simulated batch assignment for demonstration purposes and is explicitly labeled as such in every output. It is currently hidden from the dashboard's default view to avoid the misleading impression of real production grouping, though the underlying capability remains functional.

---

## 9. Path Forward

* **Defect-subtype validation**: acquire or label a dataset with real defect-subtype ground truth to validate or replace the current heuristic characterization.
* **Root-cause validation**: domain-expert review of a sample of generated explanations, and/or connection to real historical process data, to assess real-world accuracy rather than internal consistency alone.
* **Edge deployment**: benchmark on target edge hardware (e.g. Jetson-class devices) before making any edge-deployment performance claims; MobileNetV2 is the stronger architecture candidate for that phase.
* **Real trend data**: once real batch/line/timestamp metadata is available from an actual production line, the existing Trend Agent logic can be connected to it directly.
* **Observability**: the current system logs a persistent audit trail per inspection and includes 66 automated tests, but does not yet include production-grade LLM call tracing, latency monitoring, or centralized dashboards — worth adding before any pilot deployment.