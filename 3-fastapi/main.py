from fastapi import FastAPI, Header
from typing import Optional
import redis
from os import environ
import pandas as pd
from io import BytesIO
from starlette.responses import StreamingResponse
import matplotlib.pyplot as plt


app = FastAPI()
REDISHOST = environ.get("REDISHOST", None)
if REDISHOST:
    r = redis.StrictRedis(host=REDISHOST, port=6379)
else:
    r = None

MODEL_NAME = environ.get("MODEL_NAME", "A")
if MODEL_NAME == "A":
    import numpy as np

    def predict(ua=None):
        ua_proba = np.random.rand()  # uniformal random [0.0, 1.0)
        return "PC" if ua_proba > 0.5 else "Mobile"
elif MODEL_NAME == "B":
    import fasttext

    model = fasttext.load_model(environ.get('MODEL_PATH', '/models/ua_classifier.bin'))
    def predict(ua):
        ua_pred_label, _ = model.predict(ua)
        return "PC" if ua_pred_label[0] == "__label__pc" else "Mobile"
else:
    raise NotImplementedError

@app.get("/")
async def root(user_agent: Optional[str] = Header(None)):
    ua_pred = predict(user_agent)
    # Save UA to redis
    if r:
        # r.sadd('ua_true', user_agent)
        r.rpush('ua_pred', ua_pred)
    return dict(model_name=MODEL_NAME, ua_pred=f"Are you browsing on {ua_pred}?", ua_true=f"{user_agent}")

@app.get("/stats")
async def stats():
    # fetch all ua_preds
    ua_preds = [ua.decode() for ua in r.lrange("ua_pred", 0, -1)]
    plt.figure()
    graph = pd.Series(ua_preds).hist()
    buf = BytesIO()
    graph.get_figure().savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/reset")
async def reset():
    record_size_before = len(r.lrange("ua_pred", 0, -1))
    r.delete("ua_pred")
    record_size_after = len(r.lrange("ua_pred", 0, -1))
    return dict(reset_records=f"{record_size_before} to {record_size_after}.")

