import json
import re
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from hvss_calc import Hvss, HvssBaseResult

# HVSS core calculator implementation class to actually calculate the score from provided vector
calc: Hvss = Hvss()

app = FastAPI(
    title="HVSS v1.0",
    description="Healthcare Vulnerability Scoring System (HVSS) Version 1.0 Calculator.",
    version="1.0"
)


class AnalyzeRequest(BaseModel):
    description: str


@app.get("/score")
async def get_score(vector: str):
    # print(f'DEBUG:\t  Received vector: "{vector}"')
    hvss_result: HvssBaseResult = calc.calculate(vector)
    print(f'DEBUG:  Sending Response:\n{json.dumps(hvss_result, default=lambda o: o.__dict__, indent=2)}')
    return hvss_result


@app.post("/analyze")
async def analyze_description(req: AnalyzeRequest):
    """Analyze a vulnerability description and suggest HVSS metric values."""
    text = req.description.lower()

    def has(patterns):
        return any(re.search(p, text) for p in patterns)

    # --- Attack Vector (AV) ---
    if has([r'\bphysical\b', r'\busb\b', r'\bhardware\b']):
        av = 'P'
    elif has([r'\blocal\b', r'\blocal access\b', r'\buser session\b']):
        av = 'L'
    elif has([r'\badjacent\b', r'\blan\b', r'\bbluetooth\b', r'\bwi-?fi\b', r'\blocal.?network\b']):
        av = 'A'
    else:
        av = 'N'

    # --- Extended Attack Complexity (EAC) ---
    if has([r'\bextremely complex\b', r'\bextreme complexity\b']):
        eac = 'E'
    elif has([r'\bvery complex\b', r'\bmultiple conditions\b', r'\bcritical complexity\b']):
        eac = 'C'
    elif has([r'\brace condition\b', r'\bspecific config\b', r'\bhigh complexity\b', r'\bcomplex\b']):
        eac = 'H'
    elif has([r'\bmoderate complexity\b', r'\bmedium complexity\b']):
        eac = 'M'
    elif has([r'\beasily exploit\b', r'\bsimple\b', r'\blow complexity\b', r'\btrivial\b']):
        eac = 'L'
    else:
        eac = 'N'

    # --- Privileges Required (PR) ---
    if has([r'\badmin\b', r'\badministrator\b', r'\broot\b', r'\belevated\b', r'\bhigh privilege\b']):
        pr = 'H'
    elif has([r'\bauthenticated\b', r'\blow privilege\b', r'\bstandard user\b', r'\blogged.in\b']):
        pr = 'L'
    else:
        pr = 'N'

    # --- User Interaction (UI) ---
    if has([r'\buser interaction\b', r'\bclick\b', r'\bsocial engineering\b', r'\bphishing\b',
            r'\bopen.{0,10}(file|link|attachment)\b', r'\bvisit\b']):
        ui = 'R'
    else:
        ui = 'N'

    # --- Determine Impact Type ---
    is_patient = has([r'\bpatient\b', r'\bsafety\b', r'\bclinical\b', r'\bmedical device\b',
                      r'\blife.?threaten\b', r'\bharm\b', r'\binjury\b', r'\bdosage\b'])
    is_data = has([r'\bpii\b', r'\bphi\b', r'\bpersonal data\b', r'\brecords?\b', r'\bhipaa\b',
                   r'\bdata breach\b', r'\bsensitive data\b', r'\bexposure of.{0,20}data\b'])
    is_breach = has([r'\bhospital\b', r'\bnetwork access\b', r'\bimpersonat\b', r'\blateral\b',
                     r'\bpivot\b', r'\bbreach\b'])

    result = {'AV': av, 'EAC': eac, 'PR': pr, 'UI': ui}

    # --- Determine primary impact type ---
    if is_patient:
        result['XIT'] = 'XPS'
    elif is_data:
        result['XIT'] = 'XSD'
    elif is_breach:
        result['XIT'] = 'XHB'
    else:
        result['XIT'] = 'XCIA'

    # --- Always analyze all impact types ---

    # XCIA: Confidentiality
    if has([r'\bconfidentiality\b.*\bhigh\b', r'\bfull.{0,10}read\b', r'\bcomplete.{0,10}disclosure\b']):
        result['C'] = 'H'
    elif has([r'\bdata leak\b', r'\binformation disclosure\b', r'\bread\b', r'\bconfidential\b',
              r'\bexposure\b', r'\bconfidentiality\b']):
        result['C'] = 'L'
    else:
        result['C'] = 'N'
    # XCIA: Integrity
    if has([r'\bintegrity\b.*\bhigh\b', r'\bfull.{0,10}(write|control)\b', r'\bcomplete.{0,10}(modif|tamper)\b']):
        result['I'] = 'H'
    elif has([r'\bmodif\b', r'\btamper\b', r'\bwrite\b', r'\binjection\b', r'\bcorrupt\b',
              r'\bintegrity\b', r'\bxss\b', r'\bsql\b']):
        result['I'] = 'L'
    else:
        result['I'] = 'N'
    # XCIA: Availability
    if has([r'\bavailability\b.*\bhigh\b', r'\bcomplete.{0,10}(dos|denial|shutdown)\b']):
        result['A'] = 'H'
    elif has([r'\bdenial of service\b', r'\bdos\b', r'\bcrash\b', r'\bavailability\b',
              r'\bunresponsive\b', r'\bhang\b', r'\bdown\b']):
        result['A'] = 'L'
    else:
        result['A'] = 'N'

    # XPS: Patient Safety
    if has([r'\bcritical\b', r'\blife.?threaten\b', r'\bdeath\b']):
        result['XPS'] = 'C'
    elif has([r'\bmajor\b', r'\bsevere\b', r'\bserious\b']):
        result['XPS'] = 'MJ'
    elif has([r'\bmoderate\b']):
        result['XPS'] = 'MD'
    elif has([r'\blimited\b', r'\bminor\b', r'\blow\b']):
        result['XPS'] = 'L'
    else:
        result['XPS'] = 'N'

    # XSD: Sensitive Data
    if has([r'\bprimary\b']) and has([r'\bgreater\b', r'\b10.?000\b', r'\blarge\b']):
        result['XSD'] = 'PG'
    elif has([r'\bsecondary\b']) and has([r'\bgreater\b', r'\b10.?000\b', r'\blarge\b']):
        result['XSD'] = 'SG'
    elif has([r'\bprimary\b']):
        result['XSD'] = 'PL'
    elif has([r'\bsecondary\b']):
        result['XSD'] = 'SL'
    elif has([r'\bpii\b', r'\bphi\b', r'\bpersonal data\b', r'\bhipaa\b', r'\bdata breach\b',
              r'\bsensitive data\b']):
        result['XSD'] = 'PL'
    else:
        result['XSD'] = 'N'

    # XHB: Hospital Breach
    if has([r'\bimpersonat\b']):
        result['XHB'] = 'UI'
    elif has([r'\bnetwork access\b', r'\blateral\b', r'\bpivot\b']):
        result['XHB'] = 'NA'
    elif has([r'\bdevice\b']) and has([r'\bavailability\b', r'\boffline\b', r'\bshutdown\b']):
        result['XHB'] = 'DA'
    elif has([r'\bhospital\b', r'\bbreach\b']):
        result['XHB'] = 'NA'
    else:
        result['XHB'] = 'N'

    return result


@app.get("/simple", include_in_schema=False)
async def redirect_simple():
    return RedirectResponse(url='/simple/')


@app.get("/insights", include_in_schema=False)
async def redirect_insights():
    return RedirectResponse(url='/insights/')


@app.exception_handler(404)
async def custom_404_handler(_, __):
    return RedirectResponse("/")


app.mount("/insights", StaticFiles(directory="static/insights", html=True), name="insights")
app.mount("/simple", StaticFiles(directory="static/simple", html=True), name="simple")
app.mount("/", StaticFiles(directory="static/fancy", html=True), name="fancy")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
