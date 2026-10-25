"""Map Google Fact Check textual ratings to verdicts.
Rules, in order: mixed/partial phrases; negated-true or 'incorrect/inaccurate/untrue' -> FALSE;
explicit false vocabulary -> FALSE; 'no evidence/unproven/unsupported' -> UNCLEAR;
short ratings (<=40 chars) containing true/correct/accurate -> TRUE; anything else -> UNCLEAR."""
import re
MIX = r"mostly true|half[- ]true|\bmixed\b|mixture|\bpartly\b|partially|missing context|needs context|not the whole story|lacks context"
NEGT = r"(?:not|isn't|isn’t|n't|n’t|un)\s*(?:correct|true|accurate|right)|incorrect|inaccurate|untrue"
FAL = r"\bfalse\b|pants on fire|\bfake\b|misleading|\bwrong\b|pinocchio|not genuine|fabricated|\bhoax\b|\bscam\b|altered|manipulated|doctored|\bsatire\b|out of context|distort|unreliable"
UNS = r"no evidence|unproven|unsupported|not supported|insufficient evidence|lacks evidence|unverified|unclear|unknown|cannot be verified"
TRU = r"\btrue\b|\bcorrect\b|\baccurate\b|\byes\b|\bright\b"
def map_rating(r):
    r = (r or "").lower().strip()
    if not r: return "UNCLEAR"
    if re.search(MIX, r): return "MIXED"
    if re.search(NEGT, r): return "FALSE"
    if re.search(FAL, r): return "FALSE"
    if re.search(UNS, r): return "UNCLEAR"
    if len(r) <= 40 and re.search(TRU, r): return "TRUE"
    return "UNCLEAR"
