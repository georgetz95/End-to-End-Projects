import os

class Config(object):
    
    DEBUG = False
    TESTING = False
    
    basedir = os.path.abspath(os.path.dirname(__file__))
    
    SECRET_KEY = 'bionicman'
    
    UPLOADS = '/home/username/app/app/static/uploads'
    
    SESSION_COOKIE_SECURE = True
    DEFAULT_THEME = None
    
class DevelopmentConfig(Config):
    DEBUG = True
    SESSION_COOKIES_SECURE = False
    
class DebugConfig(Config):
    DEBUG = False
    
