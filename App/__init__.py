# -*- coding: utf-8 -*-
# @Time    : 20-2-7 下午2:52
# @Author  : zhuzhengyi
from flask import Flask, render_template
from flask_wtf.csrf import CSRFError

from App.blueprint import viewsblue
from App.extensions import bootstrap
from App.setting import DevelopConfig


def create_app():
	app = Flask(__name__)
	app.config.from_object(DevelopConfig)
	# print('this is host and port====',app.config.get("HOST"),app.config.get('PORT'),app.config.get('SERVER_NAME'))
	register_extensions(app)
	register_blueprints(app)
	register_errorhandlers(app)

	return app

def register_extensions(app):
	bootstrap.init_app(app)


def register_blueprints(app):
	app.register_blueprint(viewsblue)


def register_errorhandlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return render_template('errors/400.html'), 400

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template('errors/413.html'), 413

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500

    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        return render_template('errors/400.html', description=e.description), 500
