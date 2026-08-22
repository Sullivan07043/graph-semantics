"""Absolute roots for the causal LLM plugin package. Everything path-dependent reads
from here so the package runs from any working directory."""
import os

GS = "/data2/shuhao/semantic_interpretation/graph_semantics"
V6 = os.path.join(GS, "v6")
DISC = os.path.join(GS, "discovery")
T3 = os.path.join(GS, "task3_robotics")
T3P = os.path.join(T3, "task3_pipeline_v1")

# records produced by the frozen pipeline (stage A) and by this package
REC_V6 = os.path.join(V6, "outputs", "rec_v2")
REC_DISC = os.path.join(DISC, "outputs", "rec_v2")
REC_T3 = os.path.join(T3P, "outputs", "rec_v2")
REC_LLM = os.path.join(DISC, "outputs", "rec_v2_llm")      # historical pilot home
REC_JOINT = os.path.join(DISC, "outputs", "rec_v2_joint")  # merged T1+2 records

PKG = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(PKG, "outputs", "scores")

NAMING_MODEL = "openai/gpt-5.5"
API_URL = "https://openrouter.ai/api/v1"
