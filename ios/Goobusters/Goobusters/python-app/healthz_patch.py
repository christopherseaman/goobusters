"""Adds /healthz endpoint to the Flask app for iOS startup detection"""

def add_healthz(app):
    @app.route("/healthz")
    def healthz():
        return "ok", 200
    return app
