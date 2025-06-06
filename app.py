from fastapi import FastAPI, UploadFile, File, HTTPException

from inference import predict
import uvicorn
##
app = FastAPI()
@app.post("/predict/")
async def predict_route(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict(contents)
    return {"result": result}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)