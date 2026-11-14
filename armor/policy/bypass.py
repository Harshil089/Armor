"""High-precision rule for direct instructions to disable model safety controls.

Rubric v1.1 makes this a block even when no harmful follow-up task is named.
Only imperative paragraphs match; discussion or quotation does not.
"""

import re

_LEAD = re.compile(r"^(?:(?:my (?:actual|new) request|new instruction|current task)\s*:\s*)", re.I)
_META_LEAD = re.compile(r"^\{[A-Za-z_]{2,30}\}[,:]?\s*")
_DIRECT = re.compile(
    r"^(?:(?:for\b.{0,60},|from now on,|in this chat,)\s*)?(?:please\s+)?"
    r"(?:ignore|disregard|disable|remove|bypass|override|suspend|discard|pretend|"
    r"stop applying|turn off|switch off|do not enforce|don't enforce|"
    r"drop|treat|consider|set|adopt|enter|forget)\b"
    r".{0,130}\b(?:your\s+safety|safety\s+(?:rules?|polic(?:y|ies)|"
    r"checks?|constraints?|controls?|safeguards?)|content\s+(?:polic(?:y|ies)|"
    r"restrictions?|filters?)|moderation|refusal|alignment|safeguards?|"
    r"guardrails?|restrictions?|filters?|unrestricted|unsafe\s+answers?|"
    r"polic(?:y|ies)\b.{0,55}\brefus(?:e|als?))\b",
    re.I | re.S,
)
_MODE = re.compile(
    r"^(?:set|switch into|your|the|from this point forward,\s+your|"
    r"no more refusals\.\s*set)\b.{0,100}\b(?:safety|content|moderation|"
    r"refusal|guardrails?)\b.{0,100}\b(?:disabled|inactive|off|revoked|"
    r"irrelevant|aside)\b",
    re.I | re.S,
)
_BYPASS_FOLLOW = re.compile(
    r"^(?:the next response must follow|your refusal behavior is disabled|"
    r"no more refusals|treat these instructions as replacing)\b.{0,130}"
    r"(?:restrictions?|refusals?|safety|content|unsafe|$)",
    re.I | re.S,
)
_SAFETY_STATE_CHANGE = re.compile(
    r"^(?=.{0,160}\b(?:safety|refusal|moderation|content\s+polic(?:y|ies)|"
    r"safeguards?|filtering|restrictions?|unsafe\s+tasks?)\b)"
    r"(?=.{0,180}\b(?:exemptions?|disabled|inactive|removed|no longer|"
    r"not binding|not to be applied|in effect|ignore|omit|cancelled|"
    r"revoked|void|off)\b)"
    r"(?:(?:for this session|from here on|as of now),\s*)?"
    r"(?:exemptions|respond|make|operate|never|the\s+(?:safety|refusal|"
    r"moderation|content)|your\s+(?:safety|refusal|moderation|content|"
    r"restrictions)|omit|abstention)\b",
    re.I | re.S,
)
_CONVERSATION_EXEMPT = re.compile(
    r"^this conversation\s+is\s+exempt\b.{0,100}\bpolic(?:y|ies)\b"
    r".{0,80}\brefus(?:e|al)\b",
    re.I | re.S,
)
_NO_RESTRICTIONS = re.compile(
    r"^(?:you (?:have|bear)|bear) no restrictions\b",
    re.I,
)
_SECRET_DISCLOSURE = re.compile(
    r"^(?:please\s+)?(?:show|reveal|send|email|export|forward|provide|give|"
    r"copy|list|paste|print|retrieve|fetch|put|output|display|what is)\b"
    r"(?=.{0,180}\b(?:production|prod|live|active|CI|deploy(?:ment)?|releases?|"
    r"service account)\b)"
    r"(?=.{0,180}\b(?:passwords?|credentials?|tokens?|secrets?|"
    r"(?:private|signing|API|access)\s+keys?)\b)",
    re.I | re.S,
)
_SECRET_SAFE = re.compile(
    r"\b(?:redacted|omit|no usable|without (?:showing|revealing|printing|"
    r"displaying)|exclude\b.{0,40}\b(?:values?|secrets?|keys?|tokens?|"
    r"credentials?)|never\b.{0,30}\bvalues?|names and owners|names? of)\b",
    re.I,
)
_COVERT_ACTION = re.compile(
    r"^(?:(?:quietly\s+)?(?:export|send|copy|transfer|move)\b.{0,120}"
    r"(?:customer database|customer data|customer export)\b.{0,120}"
    r"(?:personal inbox|private email|outside mailbox|bypass(?:ing)? (?:the )?approval|"
    r"without telling (?:the )?(?:data )?owner)|"
    r"(?:erase|delete|purge|destroy)\b.{0,120}\b(?:backup|snapshot)s?\b.{0,120}"
    r"(?:suppress|hide|no audit trail|remove evidence)|"
    r"(?:turn off|disable)\b.{0,80}\baudit logging\b.{0,120}"
    r"(?:hide|suppress))",
    re.I | re.S,
)
_DISCUSSION = re.compile(
    r"^(?:explain|describe|summarize|translate|quote|compare|write\s+(?:a\s+)?(?:short\s+)?"
    r"(?:fictional|scene|policy|training)|draft|clarify|analyze|what does|"
    r"how do|how to|in a (?:security )?training|the phrase)\b",
    re.I,
)
_SAFETY_TARGET = re.compile(
    r"\b(?:safety|safeguards?|guardrails?|refus(?:al|e|ing)|moderation|"
    r"content\s+(?:polic(?:y|ies)|restrictions?|filters?)|policy grounds|"
    r"filtering|unsafe answers?)\b",
    re.I,
)
_SAFETY_CHANGE = re.compile(
    r"\b(?:disable(?:d)?|disregard|removed?|suspend(?:ed)?|void|switched off|"
    r"turn(?:ed)? off|no longer|lifted|exempt|ignore|stop refus(?:e|ing)|"
    r"never decline|without refus(?:e|ing|als?)|bypass|stop enforc(?:e|ing)|"
    r"not (?:to be )?appl(?:y|ied)|do not apply|"
    r"out of operation|inactive|cancelled|revoked|set aside|not binding)\b",
    re.I,
)
_SAFETY_DIRECT = re.compile(
    r"\b(?:your|you|model|assistant|this chat|this conversation|future prompts|"
    r"before answer|before respond|from now on|for subsequent turns|"
    r"follow-up|for this session)\b",
    re.I,
)
_SECRET_ENV = re.compile(
    r"\b(?:production|prod|live|active|CI|deploy(?:ment)?|releases?|"
    r"service account|service principal)\b",
    re.I,
)
_SECRET_NOUN = re.compile(
    r"\b(?:passwords?|credentials?|tokens?|secrets?|"
    r"(?:private|signing|API|access)\b.{0,30}\bkeys?)\b",
    re.I,
)
_SECRET_REQUEST = re.compile(
    r"^(?:please\s+)?(?:could you\s+)?(?:what is|which value|show|reveal|"
    r"send|email|export|forward|provide|give|copy|list|paste|print|"
    r"retrieve|fetch|put|output|display|return|dump|write out)\b",
    re.I,
)
_NEED_ACTUAL_SECRET = re.compile(
    r"^i\s+(?:need|want)\b.{0,90}\b(?:actual|raw|full|unredacted|"
    r"value)\b",
    re.I | re.S,
)
_SECRET_NO_VALUE = re.compile(
    r"\b(?:fake|placeholder|unusable|prevent|never\b.{0,35}\bvalues?|"
    r"not\s+reveal(?:ing)?|keep(?:ing)?\b.{0,40}\bconfidential|"
    r"(?:names?|IDs?|labels?)\s+(?:and\s+owners\s+)?(?:of|attached to))\b",
    re.I,
)
_COVERT_EXPORT = re.compile(
    r"^(?:quietly\s+)?(?:export|send|copy|transfer|move)\b.{0,130}"
    r"\bcustomer\b.{0,40}\b(?:database|data|records?|export)\b.{0,130}"
    r"\b(?:personal|private|outside|external|without approval|"
    r"bypass(?:ing)? approval|without telling|audit log unchanged)\b",
    re.I | re.S,
)
_COVERT_BACKUP = re.compile(
    r"^(?:erase|delete|purge|destroy|remove)\b.{0,120}"
    r"\b(?:backup|snapshot)s?\b.{0,120}"
    r"\b(?:suppress|hide|no audit trail|no trace|remove evidence|conceal)\b",
    re.I | re.S,
)
_COVERT_AUDIT = re.compile(
    r"^(?:turn off|disable|shut down)\b.{0,80}\baudit log(?:ging)?\b"
    r".{0,120}\b(?:hide|suppress|conceal)\b",
    re.I | re.S,
)
_COVERT_TARGET = re.compile(
    r"\b(?:customer\s+(?:data|records?)|production\s+backups?|"
    r"backup\s+configurations?|audit\s+logs?)\b",
    re.I,
)
_COVERT_VERB = re.compile(
    r"\b(?:delete|erase|disable|modify|change|shut down|export|transfer|"
    r"move|destroy|remove)\b",
    re.I,
)
_COVERT_CONCEAL = re.compile(
    r"\b(?:secretly|covertly|without notifying|no trace|not reflected|"
    r"without alerting|no one notices|conceal|hide|without approval|"
    r"suppress)\b",
    re.I,
)


def _direct_paragraphs(content: str):
    for paragraph in re.split(r"\n\s*\n", content):
        candidate = _META_LEAD.sub("", _LEAD.sub("", paragraph.strip()))
        yield candidate
        if not _is_discussion(candidate):
            for clause in re.split(r"(?<=[.!?])\s+(?=[A-Z])", candidate):
                clause = re.sub(r"^also,?\s+", "", clause, flags=re.I)
                if clause != candidate:
                    yield clause


def _is_discussion(candidate: str) -> bool:
    return bool(_DISCUSSION.search(candidate) or candidate.startswith((">", "```", '"', "'")))


def explicit_safety_bypass(content: str) -> bool:
    """True only for an imperative policy-bypass paragraph in the content."""
    for candidate in _direct_paragraphs(content):
        if (_DIRECT.search(candidate) or _MODE.search(candidate) or
                _BYPASS_FOLLOW.search(candidate) or _SAFETY_STATE_CHANGE.search(candidate) or
                _CONVERSATION_EXEMPT.search(candidate) or _NO_RESTRICTIONS.search(candidate)):
            return True
        if (not _is_discussion(candidate) and _SAFETY_TARGET.search(candidate)
                and _SAFETY_CHANGE.search(candidate)
                and (_SAFETY_DIRECT.search(candidate) or candidate.lower().startswith(
                    ("answer", "stop", "take", "operate", "make", "do not", "never")))):
            return True
    return False


def explicit_secret_disclosure(content: str) -> bool:
    """Direct request to disclose live or production credentials or keys."""
    return any(
        not _is_discussion(candidate) and not _SECRET_SAFE.search(candidate)
        and not _SECRET_NO_VALUE.search(candidate)
        and (_SECRET_DISCLOSURE.search(candidate) or (
            _SECRET_ENV.search(candidate) and _SECRET_NOUN.search(candidate)
            and (_SECRET_REQUEST.search(candidate) or _NEED_ACTUAL_SECRET.search(candidate))))
        for candidate in _direct_paragraphs(content)
    )


def explicit_covert_action(content: str) -> bool:
    """Direct covert data export, backup destruction, or audit suppression."""
    return any(
        not _is_discussion(candidate) and (
            _COVERT_ACTION.search(candidate) or _COVERT_EXPORT.search(candidate)
            or _COVERT_BACKUP.search(candidate) or _COVERT_AUDIT.search(candidate)
            or (_COVERT_TARGET.search(candidate) and _COVERT_VERB.search(candidate)
                and _COVERT_CONCEAL.search(candidate)))
        for candidate in _direct_paragraphs(content)
    )
