from dotenv import load_dotenv
import os
#Recordar hacer
#uv init
#uv venv
#source venv/bin/activate
#uv add mcp
from mcp.server.fastmcp.server import FastMCP

# Cargar variables de entorno desde .env automáticamente
load_dotenv()

mcp = FastMCP("Demo 🚀")
mcp.dependencies = ["fastmcp"]

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

MOODLE_TOKEN = "c47bfdd682d012455da9e4df68583830"
MOODLE_TOKEN_U = "96eea43121afcb795985da84967ce959"
MOODLE_URL = "https://chimuelo.fi.uba.ar/webservice/rest/server.php"

@mcp.tool()
def get_moodle_events() -> dict:
    """Obtiene los eventos próximos del usuario en Moodle usando el token y la URL del servicio web REST de Moodle."""
    import requests
    # Usar variables internas si no se pasan como argumento
    params = {
        'wstoken': MOODLE_TOKEN,
        'wsfunction': 'core_calendar_get_calendar_upcoming_view',
        'moodlewsrestformat': 'json',
    }
    response = requests.get(MOODLE_URL, params=params)
    response.raise_for_status()
    return response.json()

@mcp.tool()
def get_moodle_courses(userid: int = None) -> dict:
    """Obtiene los cursos en los que está inscripto un usuario en Moodle usando el token y la URL del servicio web REST de Moodle."""
    import requests
    params = {
        'wstoken': MOODLE_TOKEN_U,
        'wsfunction': 'core_enrol_get_users_courses',
        'moodlewsrestformat': 'json',
    }
    if userid is not None:
        params['userid'] = userid
    else:
        return {"error": "Se requiere el parámetro userid"}
    response = requests.get(MOODLE_URL, params=params)
    response.raise_for_status()
    return response.json()

@mcp.tool()
def get_moodle_userid(identifier: str, type: str = "username") -> dict:
    """
    Obtiene el userid de un usuario de Moodle usando el servicio core_user_get_users.
    
    Args:
        identifier (str): El username o email del usuario
        type (str): El tipo de identificador proporcionado, puede ser "username" o "email"
                   (por defecto "username")
    
    Returns:
        dict: La respuesta JSON del servicio Moodle
    """
    import requests
    
    # Validar el tipo de identificador
    if type not in ["username", "email"]:
        raise ValueError('El parámetro "type" debe ser "username" o "email"')
    
    params = {
        'wstoken': MOODLE_TOKEN_U,
        'wsfunction': 'core_user_get_users',
        'moodlewsrestformat': 'json',
        'criteria[0][key]': type,
        'criteria[0][value]': identifier
    }
    
    response = requests.get(MOODLE_URL, params=params)
    response.raise_for_status()
    return response.json()

def main():
    mcp.run()

if __name__ == "__main__":
    main()