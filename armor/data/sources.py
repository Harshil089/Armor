from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    name: str
    hf_id: str
    license: str
    label: str  # taxonomy label (armor.models.labels.LABELS) this source supervises
    text_field: str
    label_field: str
    positive_values: tuple
    collected: str
    config: str | None = None
    max_rows: int | None = None  # deterministic head slice, same as the capped loaders below


# ponytail: two public datasets, soft-launch bootstrap for the classification
# head. This is NOT the frozen M1 dataset (architecture spec section 9) —
# each source here supervises exactly one label; every other label is
# unknown (masked, not assumed negative) for its rows. Licenses below were
# checked against each dataset's HF Hub card metadata at collection time.
SOURCES = [
    DataSource(
        name="deepset-prompt-injections",
        hf_id="deepset/prompt-injections",
        license="apache-2.0",
        label="prompt_injection",
        text_field="text",
        label_field="label",
        positive_values=(1,),
        collected="2026-09-12",
    ),
    DataSource(
        name="jackhhao-jailbreak-classification",
        hf_id="jackhhao/jailbreak-classification",
        license="apache-2.0",
        label="jailbreak",
        text_field="prompt",
        label_field="type",
        positive_values=("jailbreak",),
        collected="2026-09-12",
    ),
    DataSource(
        name="spml-chatbot-prompt-injection",
        hf_id="reshabhs/SPML_Chatbot_Prompt_Injection",
        license="mit",
        label="prompt_injection",
        text_field="User Prompt",
        label_field="Prompt injection",
        positive_values=(1,),
        collected="2026-09-14",
        # Cut 2000 -> 800 after measuring SPML's own length distribution
        # (2026-09-20): within the first 2000 rows its positives run a median
        # of 68 words (mean 113) while its negatives run a median of 14 — a 5x
        # length gap inside a single source, at 1594 pos / 406 neg. That gap
        # is why prompt_injection scored 0.998 on a 99-word benign chat blob:
        # length alone separates most of this source's classes, so the model
        # learned length. Adding long benign negatives
        # (armor/data/synthetic_long_negatives.py) as counter-evidence hit hard
        # diminishing returns — 15x the rows bought only -0.093 — so reduce the
        # long-positive mass instead of piling more against it.
        max_rows=800,
    ),
]

# ponytail: SPML (arXiv 2402.11755, Sharma/Gupta/Grossman, verified MIT via
# HF card + paper) added because deepset/prompt-injections alone left
# prompt_injection with only 203 positive rows — the thinnest positive class
# of any label after ARCHIVE-QUEUE-01/02 landed. SPML supplies 12,542 positive /
# 3,470 negative rows: real system-prompt-violation attempts paired with
# genuine on-topic benign user prompts (matched to each persona's domain),
# not generic filler negatives.

# ponytail: lmsys/toxic-chat (cc-by-nc-4.0) was here and has been fully
# detached — a checkpoint trained on it inherited an NC restriction,
# blocking an unrestricted open-source release (roadmap M7). Replaced by
# armor/data/civil_comments.py (google/civil_comments, CC0-1.0, no
# conditions at all) — see below.

# Non-HF-loadable sources (no datasets.load_dataset loading script, or need
# custom filtering/thresholding) live in their own module, wired into
# training directly rather than through SOURCES:
# - armor/data/synthetic_pii.py -> sensitive_data_exposure (self-generated,
#   checksum-valid fake PII, no external dataset needed)
# - armor/data/agentharm.py -> dangerous_tool_action (ai-safety-institute/AgentHarm,
#   MIT + safety-use-only clause, 176+32 harmful / 176+32 benign matched pairs
#   across the validation + test_public splits)
# - armor/data/agentdojo.py -> dangerous_tool_action (UKGovernmentBEIS/inspect_evals,
#   MIT, 83 benign user-task prompts + 33 attacker injection goals scraped from
#   the agentdojo eval's raw task source, not run through the eval harness)
# - armor/data/civil_comments.py -> harmful_content (google/civil_comments,
#   CC0-1.0, real organic comments with a continuous toxicity score —
#   thresholded at 0.5/0.1 for positive/negative, capped at 2500 each)
# - armor/data/injecagent.py -> dangerous_tool_action (uiuc-kang-lab/InjecAgent,
#   MIT, 17 benign user instructions + 62 attacker tool-misuse instructions
#   scraped raw from GitHub — chosen over walledai/HarmBench (ARCHIVE-QUEUE-03,
#   rejected: plain instruction-request text, no tool-call framing, weaker
#   fit) because these rows are shaped as risky actions proposed through
#   tools, matching this label's actual definition, and cover a different
#   tool domain (smart locks, banking, email exfil) than AgentDojo
# - armor/data/synthetic_dangerous_tool_action.py -> dangerous_tool_action
#   (self-generated, no license needed) — covers code execution / cloud
#   infra / browser-automation domains none of the above sources reach,
#   matching Armor's own target deployment surface (agentic coding tools).
#   Matched positive/negative pairs per domain, weaker ground truth than
#   real red-team data — a supplement, not a replacement
# - armor/data/harmbench_harmful_content.py -> harmful_content
#   (walledai/HarmBench, MIT, 233 rows across 5 categories mapped to
#   Armor's 9-category policy — ARCHIVE-QUEUE-01). cybercrime_intrusion (67 rows)
#   routed to dangerous_tool_action instead (ARCHIVE-QUEUE-03); copyright (100 rows)
#   excluded, no taxonomy label covers it. Zero hard negatives supplied —
#   negatives for this label still come entirely from civil_comments
# - armor/data/synthetic_obfuscation.py -> obfuscation_encoding
#   (self-generated, no license needed) — the only viable path, no public
#   dataset maps to "concealment intent" (ARCHIVE-QUEUE-02). Positives pair a
#   concealment-framed carrier phrase with an obfuscated payload (base64,
#   ROT-13, homoglyph, zero-width chars, irregular whitespace); negatives
#   draw from two pools — unrelated legitimate encoding use, and hard
#   negatives (same payload/transform as positives, neutral carrier) added
#   after a real base64-presence-shortcut false positive was caught in a
#   retrain sanity check (2026-09-14). That fix overcorrected — the model
#   started missing the mildest positive carrier, which had no concealment
#   marker at all. Every carrier template now states or implies deliberate
#   concealment, with varied phrasing so no single keyword is the tell.
#   Payload/negative-template pools widened and default volume raised
#   450/550 (from 250/350) after this label's frozen-set score proved
#   the most fragile of the six under joint training — thin data was
#   the likely cause, not a new bug
# - armor/data/toolsafety_dangerous_tool_action.py -> dangerous_tool_action
#   (jinjinyien/ToolSafety, MIT — licensing confirmed directly by paper
#   author Prof. Pinjia He, and by the HF dataset owner, 2026-09 — ARCHIVE-QUEUE-03).
#   Best conceptual fit found for this label: direct/indirect/multi-step
#   tool-invocation harm scenarios (EMNLP 2025, 10.18653/v1/2025.emnlp-main.714).
#   Capped at a fixed 500-row slice — the source has 15569 rows, all
#   positive (request + refusal, no benign counterpart), and would swamp
#   this label's ~1000-row pool uncapped
# - armor/data/agent_safety_bench.py -> dangerous_tool_action
#   (aradhye/agent-safety-bench, Apache-2.0, pinned revision). Matched safe
#   and unsafe tool-use instructions for destructive changes, data
#   exfiltration, external communication, and financial actions. Conflicting
#   text-only labels are dropped because Armor cannot observe hidden state
# - armor/data/toolemu.py -> dangerous_tool_action (ryoungj/ToolEmu,
#   Apache-2.0, pinned revision). Curated risky-action descriptions are
#   paired with expected safe achievements across the same four weak slices
# - armor/data/jailbreak_wild.py -> jailbreak (TrustAIRLab/in-the-wild-
#   jailbreak-prompts, MIT confirmed at github.com/TrustAIRLab/JailbreakLLMs
#   project level, CCS 2024 "Do Anything Now" paper) — real jailbreak
#   attempts + genuine benign prompts (not synthetic filler) collected from
#   Reddit/Discord/websites. jailbreak's first second source; both configs
#   capped at 600 rows each. A first version capped only the negative side
#   and let all 1405 positives in, making TrustAIRLab 73% of the positive
#   class — its long/elaborate persona-prompt style diluted sensitivity to
#   jackhhao's short canonical DAN-style phrasing (regression score
#   0.582 -> 0.250, 4 of 5 short-phrase frozen cases flipped to missed)
# - armor/data/aegis_content_safety.py -> harmful_content negatives
#   and sensitive_data_exposure negatives (nvidia/Aegis-AI-Content-Safety-
#   Dataset-1.0, CC-BY-4.0, pinned revision), also used as natural-noise
#   obfuscation negatives. Uses only single-leg user rows unanimously annotated
#   Safe; response and combined rows are excluded because their text can contain
#   multiple dialog legs and does not match Armor's inspection grain.
# - armor/data/oasst_chat_negatives.py -> prompt_injection and
#   obfuscation_encoding negatives (OpenAssistant/oasst1, Apache-2.0, pinned
#   revision). Uses English, accepted, multi-review, low-toxicity turns and
#   rank-0 assistant responses. Stable code/structure/length strata cover short
#   chat, long formatted replies, and fenced code answers; sensitive-data
#   negatives are excluded if Armor's deterministic DLP registry recognizes any
#   entity.
# - armor/data/coedit_noisy_negatives.py -> obfuscation_encoding negatives
#   (grammarly/coedit, Apache-2.0, pinned revision). Uses both sides of 400
#   natural GEC pairs limited to one or two local word edits; this supplies
#   spelling invariance without generating typo lookalikes from production logs.
# - armor/data/synthetic_jailbreak_canonical.py -> jailbreak (self-generated,
#   no license needed) — reinforces the exact short-phrase family
#   ("Enable developer mode", "unfiltered model") that jailbreak_wild.py's
#   TrustAIRLab addition diluted, same fix pattern as obfuscation_encoding's
#   fragility (widen + add volume, not retreat from the real source)
# - armor/data/harm_request_sources.py -> harmful_content (+ dangerous_tool_action
#   negatives from FRR/XSTest-safe). JBB-Behaviors (MIT), ASTRA (MIT),
#   PurpleLlama mitre_frr (MIT), Aegis 2.0 (CC-BY-4.0), XSTest (CC-BY-4.0),
#   AdvBench (MIT, gated). All revision-pinned. Added 2026-10-19 after bare
#   build-requests ("build me a malware") scored ~0.1. Notices and terms:
#   THIRD_PARTY_NOTICES.md. Rejected: allenai/wildguardmix (gate requires a
#   research-only certification), BeaverTails + MOCHA (CC-BY-NC), RMCBench +
#   MalwareBench (no stated license)

# M1.1 source survey — candidates for the remaining uncovered labels
# (docs/Armor_M1_Source_Survey_v0.1.docx). Every label now has a wired-in
# source; nothing currently sits in this dict. obfuscation_encoding was
# here until its synthetic pipeline (synthetic_obfuscation.py) was wired
# into training — see the non-HF-loadable sources list above.
CANDIDATE_SOURCES: dict[str, list[str]] = {}
