"""Generates a submission document (HTML & print-ready PDF format) for the student.

Includes all required checklist fields:
1. GitHub Repository URL with access granted to ai.assignments@wacpinternational.org
2. Student's Full Name
3. Candidate Registration Number
4. CRS Name: Artificial Intelligence
5. Course Name
6. School Name
7. Functional Streamlit Web App link
8. Technical project summary and simulation results
"""
import argparse
from pathlib import Path

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ParkVision AI - Project Submission Document</title>
<style>
    @page { size: A4; margin: 20mm; }
    body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #222;
        line-height: 1.5;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .header {
        border-bottom: 3px solid #1a73e8;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }
    h1 { margin: 0 0 6px 0; color: #1a73e8; font-size: 24px; }
    .subtitle { color: #555; font-size: 14px; margin-bottom: 12px; }
    .meta-box {
        background: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        row-gap: 10px;
        column-gap: 20px;
        font-size: 14px;
    }
    .meta-item strong { color: #333; }
    .badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 13px;
    }
    .section-title {
        color: #202124;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 6px;
        margin-top: 24px;
        font-size: 17px;
    }
    .link-box {
        background: #e6f4ea;
        border: 1px solid #ceead6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 16px 0;
        font-size: 14px;
    }
    .link-box a { color: #137333; font-weight: bold; word-break: break-all; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #dadce0;
        padding: 8px 12px;
        text-align: left;
    }
    th { background: #f1f3f4; color: #333; }
    .footer {
        margin-top: 40px;
        padding-top: 12px;
        border-top: 1px solid #eee;
        font-size: 12px;
        color: #777;
        text-align: center;
    }
    @media print {
        body { padding: 0; }
        .no-print { display: none; }
    }
</style>
</head>
<body>

<div class="header">
    <h1>🅿️ ParkVision AI: Intelligent Urban Parking Analytics</h1>
    <div class="subtitle">Space Optimisation & Real-Time Monitoring Platform | Final Project Submission</div>
    <span class="badge">UrbanFlow AI & Municipal Analytics</span>
</div>

<div class="meta-box">
    <div class="meta-grid">
        <div class="meta-item"><strong>Student's Full Name:</strong> {student_name}</div>
        <div class="meta-item"><strong>Candidate Registration No.:</strong> {candidate_reg}</div>
        <div class="meta-item"><strong>CRS Name:</strong> <span class="badge">Artificial Intelligence</span></div>
        <div class="meta-item"><strong>Course Name:</strong> {course_name}</div>
        <div class="meta-item"><strong>School / Institution Name:</strong> {school_name}</div>
        <div class="meta-item"><strong>Submission Date:</strong> {submission_date}</div>
    </div>
</div>

<h2 class="section-title">1. Required Submission Links</h2>

<div class="link-box">
    <strong>🔗 GitHub Repository URL:</strong><br>
    <a href="{github_url}" target="_blank">{github_url}</a><br>
    <small style="color: #2b7a42;">✔ Collaborator / reviewer access granted to: <code>ai.assignments@wacpinternational.org</code></small>
</div>

<div class="link-box" style="background:#fef7e0; border-color:#fce8b2;">
    <strong style="color:#b06000;">🌐 Live Streamlit Cloud Application:</strong><br>
    <a href="{streamlit_url}" target="_blank" style="color:#b06000;">{streamlit_url}</a>
</div>

<h2 class="section-title">2. GitHub Repository Deliverables Checklist</h2>
<table>
    <tr><th>Item</th><th>Filename / Location</th><th>Status</th></tr>
    <tr>
        <td>Python Codebase</td>
        <td><code>app.py</code>, <code>logic.py</code>, <code>predictor.py</code>, <code>config.py</code></td>
        <td>✔ Complete & Functional</td>
    </tr>
    <tr>
        <td>Training Notebook</td>
        <td><code>ParkVision_AI_Training.ipynb</code> (Google Colab ready)</td>
        <td>✔ Included with Anti-Data Loss Drive Sync</td>
    </tr>
    <tr>
        <td>Trained Model Weights</td>
        <td><code>models/parkvision_mobilenetv2.keras</code></td>
        <td>✔ MobileNetV2 Fine-Tuned (ImageNet backbone)</td>
    </tr>
    <tr>
        <td>Dataset Pipeline</td>
        <td><code>prepare_dataset.py</code>, <code>dataset/</code> (70% train / 15% val / 15% test)</td>
        <td>✔ PKLot Multi-Weather Calibrated</td>
    </tr>
    <tr>
        <td>Dependencies</td>
        <td><code>requirements.txt</code></td>
        <td>✔ Configured for Streamlit Cloud & Colab</td>
    </tr>
    <tr>
        <td>Comprehensive Documentation</td>
        <td><code>README.md</code> (academic citations, benchmark comparisons, results)</td>
        <td>✔ Formatted with Literature References</td>
    </tr>
</table>

<h2 class="section-title">3. Model Architecture & Training Summary</h2>
<table>
    <tr><th>Specification</th><th>Details</th></tr>
    <tr><td>Computer Vision Technique</td><td>Crop-based classification using <strong>MobileNetV2</strong> with transfer learning</td></tr>
    <tr><td>Input Resolution</td><td>224 × 224 pixels (3 channels RGB), normalized to [-1, 1]</td></tr>
    <tr><td>Data Augmentation</td><td>Random horizontal flip, rotation (±14°), brightness (±20%), contrast (±10%)</td></tr>
    <tr><td>Phase 1 Training</td><td>15 epochs, Adam (LR = 1e-3), frozen backbone, binary cross-entropy</td></tr>
    <tr><td>Phase 2 Fine-Tuning</td><td>10 epochs, Adam (LR = 1e-5), layers ≥ 100 unfrozen, BatchNorm frozen</td></tr>
    <tr><td>Class Weighting</td><td>Applied automatically to ensure equal loss contribution across classes</td></tr>
</table>

<h2 class="section-title">4. Space Optimisation & Real-Time Parking Logic</h2>
<table>
    <tr><th>Metric / Feature</th><th>Implementation Logic</th><th>Output Range / Action</th></tr>
    <tr><td>Slot Detection</td><td>P(occupied) evaluated per slot crop against threshold (default 0.50)</td><td>Occupied (Red) / Empty (Green)</td></tr>
    <tr><td>Availability Metrics</td><td>Total slots, Occupied slots, Available free slots</td><td>Real-time numerical count cards</td></tr>
    <tr><td>Occupancy Rate</td><td>Occupancy % = (Occupied / Total) × 100</td><td>Continuous progress bar & percentage</td></tr>
    <tr><td>Congestion Classification</td><td>Low (&lt;40%), Moderate (40%–75%), High (&gt;75%)</td><td>Color-coded UI badge</td></tr>
    <tr><td>Decision Recommendation</td><td>Available == 0 → "Parking full - try another area"<br>Available &gt; 0 → "Slots available - proceed"</td><td>Actionable guidance for urban drivers</td></tr>
</table>

<div class="footer">
    ParkVision AI · Final Project Submission Document · Designed for submission to UrbanFlow AI & Academic Evaluators.
</div>

</body>
</html>
"""

def generate_doc(
    student_name="[Student Name]",
    candidate_reg="[Candidate Reg No]",
    course_name="Machine Learning and Deep Learning",
    school_name="[School / University Name]",
    github_url="https://github.com/[YourUsername]/ParkVision-AI",
    streamlit_url="https://[your-app-name].streamlit.app",
    out_file="Submission_Document.html"
):
    from datetime import date
    content = (HTML_TEMPLATE
               .replace("{student_name}", student_name)
               .replace("{candidate_reg}", candidate_reg)
               .replace("{course_name}", course_name)
               .replace("{school_name}", school_name)
               .replace("{github_url}", github_url)
               .replace("{streamlit_url}", streamlit_url)
               .replace("{submission_date}", date.today().strftime("%B %d, %Y")))
    p = Path(out_file)
    p.write_text(content, encoding="utf-8")
    print(f"[OK] Generated submission document: {p.resolve()}")
    print("     Open this file in Google Chrome / Microsoft Edge and choose 'Print' -> 'Save as PDF' to produce your final submission PDF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission document HTML/PDF")
    parser.add_argument("--name", default="Mann Patel", help="Student's full name")
    parser.add_argument("--reg", default="AI-2026-REG-01", help="Candidate registration number")
    parser.add_argument("--course", default="Artificial Intelligence & Computer Vision", help="Course name")
    parser.add_argument("--school", default="School of Computing & Artificial Intelligence", help="School name")
    parser.add_argument("--github", default="https://github.com/YourUsername/ParkVision-AI", help="GitHub repo URL")
    parser.add_argument("--streamlit", default="https://parkvision-ai.streamlit.app", help="Streamlit app URL")
    parser.add_argument("--out", default="Submission_Document.html", help="Output file path")
    args = parser.parse_args()

    generate_doc(args.name, args.reg, args.course, args.school, args.github, args.streamlit, args.out)
