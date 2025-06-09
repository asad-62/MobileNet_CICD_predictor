from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from inference import predict
from fastapi.responses import JSONResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)

    # Return JSON if client requests it (e.g., curl or script)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content={"prediction": result})

    # Else return HTML page
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "face_type": result["face_type"],
        "glasses": ", ".join(result["suggested_glasses"])

    })

