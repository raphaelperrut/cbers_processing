# -*- coding: utf-8 -*-
from __future__ import annotations

import re


class LogSummaryParser:
    RE_TIME = re.compile(r"^\[TIME\]\s+(.*?):\s+([0-9.]+s)$")
    RE_QA2 = re.compile(r"^\[(QA2:[A-Z]+)\]\s+(.*)$")
    RE_QA3 = re.compile(r"^\[(QA3:[A-Z]+)\]\s+(.*)$")
    RE_TOTAL = re.compile(r"^\[TIME\]\s+TOTAL:\s+([0-9.]+s)$")

    RE_N = re.compile(r"\|\s*n=([0-9]+)")
    RE_SKIPPED_LOW_VALID = re.compile(r"skipped_low_valid=([0-9]+)")
    RE_SKIPPED_SMALL_MASK = re.compile(r"skipped_small_mask=([0-9]+)")
    RE_GRAD_MEAN = re.compile(r"corr\(grad\([^)]+\),grad\([^)]+\)\)\s+mean=([0-9.]+)")
    RE_LAP_MEAN = re.compile(r"corr\(lap\([^)]+\),lap\([^)]+\)\)\s+mean=([0-9.]+)")
    RE_LEAK_MEAN = re.compile(r"leak corr\(grad\([^)]+\),grad\(\|CbCr\|\)\)\s+mean=([0-9.]+)")

    def __init__(self):
        self.times = {}
        self.total_time = None

        self.qa2_status = None
        self.qa2_text = None
        self.qa2_n = None
        self.qa2_grad_mean = None
        self.qa2_lap_mean = None
        self.qa2_leak_mean = None
        self.qa2_skipped_low_valid = None
        self.qa2_skipped_small_mask = None

        self.qa3_status = None
        self.qa3_text = None
        self.qa3_n = None
        self.qa3_grad_mean = None
        self.qa3_lap_mean = None
        self.qa3_leak_mean = None
        self.qa3_skipped_low_valid = None
        self.qa3_skipped_small_mask = None

        self.progress = 0

    def _extract_qa_metrics(self, text: str) -> dict:
        out = {}

        m = self.RE_N.search(text)
        if m:
            out["n"] = m.group(1)

        m = self.RE_SKIPPED_LOW_VALID.search(text)
        if m:
            out["skipped_low_valid"] = m.group(1)

        m = self.RE_SKIPPED_SMALL_MASK.search(text)
        if m:
            out["skipped_small_mask"] = m.group(1)

        m = self.RE_GRAD_MEAN.search(text)
        if m:
            out["grad_mean"] = m.group(1)

        m = self.RE_LAP_MEAN.search(text)
        if m:
            out["lap_mean"] = m.group(1)

        m = self.RE_LEAK_MEAN.search(text)
        if m:
            out["leak_mean"] = m.group(1)

        return out

    def feed_line(self, line: str):
        line = line.strip()
        if not line:
            return

        if "[PIPE] STEP0" in line:
            self.progress = max(self.progress, 5)
        elif "[TIME] STEP0" in line:
            self.progress = max(self.progress, 15)
        elif "[PIPE] STEP1 Build LR RGB VRT" in line:
            self.progress = max(self.progress, 16)
        elif "[TIME] STEP1 build LR RGB VRT" in line:
            self.progress = max(self.progress, 20)
        elif "[PIPE] STEP1b" in line:
            self.progress = max(self.progress, 22)
        elif "[TIME] STEP1b sensor LR normalize" in line:
            self.progress = max(self.progress, 30)
        elif "[PIPE] STEP2" in line:
            self.progress = max(self.progress, 35)
        elif "[TIME] STEP2 resample SENSOR guide" in line:
            self.progress = max(self.progress, 50)
        elif "[PIPE] STEP7" in line:
            self.progress = max(self.progress, 55)
        elif "[TIME] STEP7 FUSE" in line:
            self.progress = max(self.progress, 85)
        elif "[TIME] QA3" in line:
            self.progress = max(self.progress, 92)
        elif "COG light export" in line:
            self.progress = max(self.progress, 96)
        elif "[TIME] COG export" in line:
            self.progress = max(self.progress, 100)
        elif line.startswith("OK: "):
            self.progress = max(self.progress, 90)
        elif "[TIME] TOTAL:" in line:
            self.progress = 100

        m = self.RE_TIME.match(line)
        if m:
            label, value = m.groups()
            if label == "TOTAL":
                self.total_time = value
            else:
                self.times[label] = value
            return

        m = self.RE_TOTAL.match(line)
        if m:
            self.total_time = m.group(1)
            return

        m = self.RE_QA2.match(line)
        if m:
            self.qa2_status = m.group(1)
            self.qa2_text = m.group(2)
            qa = self._extract_qa_metrics(m.group(2))
            self.qa2_n = qa.get("n")
            self.qa2_grad_mean = qa.get("grad_mean")
            self.qa2_lap_mean = qa.get("lap_mean")
            self.qa2_leak_mean = qa.get("leak_mean")
            self.qa2_skipped_low_valid = qa.get("skipped_low_valid")
            self.qa2_skipped_small_mask = qa.get("skipped_small_mask")
            return

        m = self.RE_QA3.match(line)
        if m:
            self.qa3_status = m.group(1)
            self.qa3_text = m.group(2)
            qa = self._extract_qa_metrics(m.group(2))
            self.qa3_n = qa.get("n")
            self.qa3_grad_mean = qa.get("grad_mean")
            self.qa3_lap_mean = qa.get("lap_mean")
            self.qa3_leak_mean = qa.get("leak_mean")
            self.qa3_skipped_low_valid = qa.get("skipped_low_valid")
            self.qa3_skipped_small_mask = qa.get("skipped_small_mask")
            return

    def summary(self) -> dict:
        return {
            "times": dict(self.times),
            "total_time": self.total_time,

            "qa2_status": self.qa2_status,
            "qa2_text": self.qa2_text,
            "qa2_n": self.qa2_n,
            "qa2_grad_mean": self.qa2_grad_mean,
            "qa2_lap_mean": self.qa2_lap_mean,
            "qa2_leak_mean": self.qa2_leak_mean,
            "qa2_skipped_low_valid": self.qa2_skipped_low_valid,
            "qa2_skipped_small_mask": self.qa2_skipped_small_mask,

            "qa3_status": self.qa3_status,
            "qa3_text": self.qa3_text,
            "qa3_n": self.qa3_n,
            "qa3_grad_mean": self.qa3_grad_mean,
            "qa3_lap_mean": self.qa3_lap_mean,
            "qa3_leak_mean": self.qa3_leak_mean,
            "qa3_skipped_low_valid": self.qa3_skipped_low_valid,
            "qa3_skipped_small_mask": self.qa3_skipped_small_mask,

            "progress": self.progress,
        }