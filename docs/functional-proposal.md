# Agentic Visual Quality Inspection for Casting Manufacturing

## Functional Proposal

### 1. Overview

This proposal describes an **AI-based visual quality inspection system for cast metal parts**.

Instead of simply saying **"defective" or "not defective,"** the system is designed to:

* Detect the defect
* Identify the type of defect
* Suggest possible root causes
* Recommend what to do with the part
* Keep a record for traceability

The system uses a **vision model to see the defect** and different **AI agents to analyze and make decisions**.

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
* Identifies where the defect is located

### AI Agents → **Think and decide**

* Identify the defect type
* Find possible root causes
* Recommend what to do with the part
* Monitor defect trends
* Create inspection records

This separation makes the system easier to **improve, monitor, and maintain**.

---

## 4. Agent Roles

| Agent                      | What it does                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Inspection Agent**       | Analyzes the image, detects defects, gives a confidence score, and shows where the defect is located.          |
| **Characterization Agent** | Designed to identify defect type (e.g., porosity, shrinkage, crack, surface blemish) as the knowledge base and training data mature. |
| **Root-Cause Agent**       | Finds possible process-related causes using a defect knowledge base.                                           |
| **Disposition Agent**      | Recommends whether the part should be accepted, reworked, or scrapped. Sends uncertain cases for human review. |
| **Trend Agent**            | Monitors defect patterns across batches and production lines to identify process problems early.               |
| **Reporting Agent**        | Creates a complete inspection record with defect details, root cause, decision, and image evidence.            |
| **Orchestrator Agent**     | Coordinates all the agents and manages the overall workflow and human escalation.                              |

---

## 5. How the System Works

### Step 1: Image Inspection

A camera captures the image of the cast part.

The **Inspection Agent** analyzes the image and detects any defect.

---

### Step 2: Defect Identification

The **Characterization Agent** is designed to identify the type of defect as the system matures, for example:

* Porosity
* Shrinkage
* Crack
* Surface blemish

*(Note: initial detection is validated as a binary outcome — defective / not defective. Distinguishing between specific defect types is a planned capability, dependent on a defect-labeled dataset — see Section 7.)*

---

### Step 3: Root Cause Analysis

The **Root-Cause Agent** checks the knowledge base and suggests possible causes.

For example, a defect may be related to:

* Incorrect mold temperature
* Improper pouring speed
* Cooling problems

The agent uses available knowledge instead of simply guessing.

---

### Step 4: Decide What to Do

The **Disposition Agent** recommends:

* **Accept**
* **Rework**
* **Scrap**

If the system is not confident, it sends the case to a **human inspector**.

---

### Step 5: Monitor Defect Trends

The **Trend Agent** monitors defects across:

* Batches
* Production lines
* Time periods

This can help identify **process drift early**, before it becomes a major quality problem.

---

### Step 6: Create an Inspection Record

The **Reporting Agent** creates a structured record containing:

* Defect type
* Possible root cause
* Recommended action
* Image evidence
* Inspection result

This provides proper **traceability**.

---

### Step 7: Coordinate Everything

The **Orchestrator Agent** manages the complete process.

It:

* Coordinates all agents
* Manages the sequence of activities
* Handles exceptions
* Sends uncertain cases to human reviewers

---

## 6. Expected Outcomes

The system can provide:

* **More consistent inspection** across different shifts and inspectors.
* **Faster detection of process problems** before they affect production yield.
* **Better traceability** of defects and inspection decisions.
* **A history of defects** that can be connected to manufacturing process variables.
* **Human involvement in uncertain cases**, instead of allowing AI to make every decision automatically.

---

## 7. Current Validation Status

* **Defect detection (binary):** Validated on a real, labeled image dataset of cast parts.
* **Inference speed:** Validated on GPU hardware (NVIDIA Titan RTX), with per-image processing times consistent with production line cycle-time requirements. This establishes technical feasibility and serves as the foundation for future deployment on edge GPU hardware (e.g., NVIDIA Jetson or line-side embedded inference devices), enabling on-site, real-time inspection without dependency on centralized or cloud compute.
* **Defect-type characterization, root-cause reasoning, and trend monitoring:** Architected and designed, but dependent on additional data — a defect-labeled dataset for characterization, a curated knowledge base for root-cause reasoning, and historical batch/line data for trend detection. These are planned next-phase additions, not yet trained or validated capabilities.

---

## 8. Scope and Requirements

This proposal describes the **functional design** of the system.

The system requires:

### 1. Image Dataset

A labeled dataset of cast part images, including:

* Defective parts
* Non-defective parts

This will be used to develop the vision model.

*(A defect-subtype-labeled dataset — one that identifies porosity vs. crack vs. shrinkage individually — would be required to fully realize defect-type characterization. The current dataset supports binary detection only.)*

### 2. Defect Knowledge Base

A knowledge base connecting:

**Defect → Possible Process Causes**

For example:

**Porosity → Possible casting process causes**

The knowledge base can gradually improve as more inspection data and manufacturing history become available.

