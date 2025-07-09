# Notas importantes para el uso del MCP Playwright

- **Modo headless obligatorio:**
  - El MCP de Playwright debe ejecutarse siempre en modo `headless=True` (sin interfaz gráfica), ya que de lo contrario fallará por falta de XServer en la mayoría de los entornos Linux/servidores.
  - El cliente fuerza este parámetro automáticamente, pero si modificás el código o usás otro cliente, asegurate de que el parámetro `headless` esté presente en los argumentos.
  - Si usás prompts o configuraciones personalizadas, verificá que las URLs sean completas (con `https://`).

- **Integración A2A:**
  - Para que este MCP agent pueda ser consultado por otros agentes (A2A, agent-to-agent), asegurate de exponer el endpoint o mecanismo de comunicación adecuado según tu arquitectura.
  - Revisa la documentación de tu framework MCP para detalles sobre cómo registrar y exponer agentes para consultas externas.

---

## MCP Servers disponibles (ver `mcp_config.json`)

- **playwright**: Automatización de navegador (web scraping, testing, etc.) usando Playwright. Se ejecuta vía `npx @executeautomation/playwright-mcp-server --headless:True`.
- **mcp-atlassian**: Acceso a APIs de Atlassian (Confluence, Jira) usando un contenedor Docker (`ghcr.io/sooperset/mcp-atlassian:latest`).
- **moodle-mcp**: Servidor local para integración con Moodle, ejecuta `server.py` (ver código en `server.py`).

## Clientes disponibles

- **client.py**: Cliente base MCP, integración con memoria y herramientas.
- **client_good.py**: Cliente con manejo detallado de herramientas, respuestas y debugging.
- **client_memory.py**: Cliente con memoria de conversación persistente en SQLite (`conversation_memory.db`).

## Detalle de la memoria (memory)

- Se utiliza una clase `SimpleMemory` (ver en los clientes) que almacena cada conversación (mensaje de usuario, respuesta AI, timestamp, session_id) en una base SQLite local (`conversation_memory.db`).
- Permite recuperar el historial reciente de cada sesión para dar contexto a la IA.
- No utiliza embeddings ni búsqueda semántica, solo almacenamiento y recuperación simple por sesión.

---

Para más detalles, revisar los archivos de código y la configuración en `mcp_config.json`.
