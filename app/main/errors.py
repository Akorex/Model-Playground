from app import app


@app.errorhandler(404)
def page_not_found(e):
    return "<h1> Page not found </h1>"