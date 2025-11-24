"""Web-based pattern review interface."""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import List, Optional

import click

from talk2metadata.core.qa_generation.patterns import PathPattern
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class PatternReviewHandler(BaseHTTPRequestHandler):
    """HTTP handler for pattern review web interface."""

    patterns_file: Optional[Path] = None
    patterns_data: Optional[dict] = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = self._generate_html()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/api/patterns":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(self.patterns_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests (save patterns)."""
        if self.path == "/api/save":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                # Save to file
                if self.patterns_file:
                    with open(self.patterns_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved patterns to {self.patterns_file}")
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                else:
                    self.send_response(500)
                    self.end_headers()
            except Exception as e:
                logger.error(f"Failed to save patterns: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                )
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _generate_html(self) -> str:
        """Generate HTML for pattern review interface."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Path Pattern Review</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: #0366d6;
            color: white;
        }}
        .btn-primary:hover {{
            background: #0256c2;
        }}
        .btn-success {{
            background: #28a745;
            color: white;
        }}
        .btn-success:hover {{
            background: #218838;
        }}
        .btn-danger {{
            background: #dc3545;
            color: white;
        }}
        .btn-danger:hover {{
            background: #c82333;
        }}
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        .btn-secondary:hover {{
            background: #5a6268;
        }}
        .patterns-list {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .pattern-card {{
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 20px;
            background: #fafafa;
            transition: all 0.2s;
        }}
        .pattern-card:hover {{
            border-color: #0366d6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .pattern-card.editing {{
            border-color: #0366d6;
            background: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
        .pattern-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }}
        .pattern-info {{
            flex: 1;
        }}
        .pattern-path {{
            font-size: 16px;
            font-weight: 600;
            color: #0366d6;
            margin-bottom: 5px;
        }}
        .pattern-path input {{
            font-size: 16px;
            font-weight: 600;
            color: #0366d6;
            border: 1px solid #ddd;
            padding: 5px 10px;
            border-radius: 4px;
            width: 100%;
        }}
        .pattern-semantic {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .pattern-semantic textarea {{
            width: 100%;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 4px;
            font-size: 14px;
            resize: vertical;
            min-height: 60px;
        }}
        .pattern-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .detail-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}
        .detail-label {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .detail-value {{
            font-size: 14px;
            color: #333;
        }}
        .detail-value input,
        .detail-value textarea,
        .detail-value select {{
            width: 100%;
            border: 1px solid #ddd;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 14px;
        }}
        .detail-value textarea {{
            resize: vertical;
            min-height: 80px;
        }}
        .pattern-actions {{
            display: flex;
            gap: 5px;
        }}
        .pattern-actions button {{
            padding: 6px 12px;
            font-size: 12px;
        }}
        .add-pattern {{
            border: 2px dashed #ddd;
            border-radius: 6px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: white;
        }}
        .add-pattern:hover {{
            border-color: #0366d6;
            background: #f0f7ff;
        }}
        .status-message {{
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            display: none;
        }}
        .status-message.success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status-message.error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .status-message.show {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Path Pattern Review</h1>
        <p class="subtitle">Review and edit path patterns for QA generation</p>

        <div id="status" class="status-message"></div>

        <div class="toolbar">
            <button class="btn-success" onclick="addPattern()">+ Add Pattern</button>
            <button class="btn-primary" onclick="savePatterns()">💾 Save Changes</button>
            <button class="btn-secondary" onclick="loadPatterns()">🔄 Reload</button>
        </div>

        <div id="patterns-list" class="patterns-list"></div>
    </div>

    <script>
        let patterns = [];
        let editingIndex = null;

        async function loadPatterns() {{
            try {{
                const response = await fetch('/api/patterns');
                const data = await response.json();
                patterns = data.patterns || [];
                renderPatterns();
                showStatus('Patterns loaded successfully', 'success');
            }} catch (error) {{
                showStatus('Failed to load patterns: ' + error.message, 'error');
            }}
        }}

        function renderPatterns() {{
            const container = document.getElementById('patterns-list');
            container.innerHTML = '';

            patterns.forEach((pattern, index) => {{
                const card = document.createElement('div');
                card.className = 'pattern-card' + (editingIndex === index ? ' editing' : '');

                const isEditing = editingIndex === index;
                const pathStr = pattern.pattern.join(' → ');
                const pathInput = isEditing ?
                    '<input type="text" value="' + pattern.pattern.join(' -> ') + '" ' +
                    'onchange="updatePattern(' + index + ', \\'pattern\\', this.value.split(\\' -> \\').map(s => s.trim()))">' :
                    pathStr;

                const semanticInput = isEditing ?
                    '<textarea onchange="updatePattern(' + index + ', \\'semantic\\', this.value)">' +
                    (pattern.semantic || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</textarea>' :
                    (pattern.semantic || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');

                const actions = isEditing ?
                    '<button class="btn-success" onclick="saveEdit(' + index + ')">✓ Save</button>' +
                    '<button class="btn-secondary" onclick="cancelEdit()">✗ Cancel</button>' :
                    '<button class="btn-primary" onclick="editPattern(' + index + ')">✏️ Edit</button>' +
                    '<button class="btn-danger" onclick="deletePattern(' + index + ')">🗑️ Delete</button>';

                const templateInput = isEditing ?
                    '<textarea onchange="updatePattern(' + index + ', \\'question_template\\', this.value)">' +
                    (pattern.question_template || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</textarea>' :
                    '<pre style="margin:0; white-space: pre-wrap;">' +
                    (pattern.question_template || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</pre>';

                const difficultySelect = isEditing ?
                    '<select onchange="updatePattern(' + index + ', \\'difficulty\\', this.value)">' +
                    '<option value="easy"' + (pattern.difficulty === 'easy' ? ' selected' : '') + '>Easy</option>' +
                    '<option value="medium"' + (pattern.difficulty === 'medium' ? ' selected' : '') + '>Medium</option>' +
                    '<option value="hard"' + (pattern.difficulty === 'hard' ? ' selected' : '') + '>Hard</option>' +
                    '</select>' :
                    pattern.difficulty || 'medium';

                const answerTypeSelect = isEditing ?
                    '<select onchange="updatePattern(' + index + ', \\'answer_type\\', this.value)">' +
                    '<option value="single"' + (pattern.answer_type === 'single' ? ' selected' : '') + '>Single</option>' +
                    '<option value="multiple"' + (pattern.answer_type === 'multiple' ? ' selected' : '') + '>Multiple</option>' +
                    '<option value="aggregate"' + (pattern.answer_type === 'aggregate' ? ' selected' : '') + '>Aggregate</option>' +
                    '</select>' :
                    pattern.answer_type || 'multiple';

                const descInput = isEditing ?
                    '<textarea onchange="updatePattern(' + index + ', \\'description\\', this.value)">' +
                    (pattern.description || '').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</textarea>' :
                    (pattern.description || '<em>No description</em>');

                card.innerHTML =
                    '<div class="pattern-header">' +
                        '<div class="pattern-info">' +
                            '<div class="pattern-path">' + pathInput + '</div>' +
                            '<div class="pattern-semantic">' + semanticInput + '</div>' +
                        '</div>' +
                        '<div class="pattern-actions">' + actions + '</div>' +
                    '</div>' +
                    '<div class="pattern-details">' +
                        '<div class="detail-group">' +
                            '<div class="detail-label">Question Template</div>' +
                            '<div class="detail-value">' + templateInput + '</div>' +
                        '</div>' +
                        '<div class="detail-group">' +
                            '<div class="detail-label">Difficulty</div>' +
                            '<div class="detail-value">' + difficultySelect + '</div>' +
                        '</div>' +
                        '<div class="detail-group">' +
                            '<div class="detail-label">Answer Type</div>' +
                            '<div class="detail-value">' + answerTypeSelect + '</div>' +
                        '</div>' +
                        '<div class="detail-group">' +
                            '<div class="detail-label">Description</div>' +
                            '<div class="detail-value">' + descInput + '</div>' +
                        '</div>' +
                    '</div>';
                container.appendChild(card);
            }});

            // Add "Add Pattern" button
            const addBtn = document.createElement('div');
            addBtn.className = 'add-pattern';
            addBtn.innerHTML = '<div style="font-size: 18px; color: #666;">+ Click to add new pattern</div>';
            addBtn.onclick = addPattern;
            container.appendChild(addBtn);
        }}

        function editPattern(index) {{
            editingIndex = index;
            renderPatterns();
        }}

        function saveEdit(index) {{
            editingIndex = null;
            renderPatterns();
            showStatus('Pattern updated', 'success');
        }}

        function cancelEdit() {{
            editingIndex = null;
            renderPatterns();
        }}

        function updatePattern(index, field, value) {{
            if (field === 'pattern' && Array.isArray(value)) {{
                patterns[index].pattern = value;
            }} else {{
                patterns[index][field] = value;
            }}
        }}

        function deletePattern(index) {{
            if (confirm('Are you sure you want to delete this pattern?')) {{
                patterns.splice(index, 1);
                renderPatterns();
                showStatus('Pattern deleted', 'success');
            }}
        }}

        function addPattern() {{
            const newPattern = {{
                pattern: ['table1', 'wamex_reports'],
                semantic: 'New pattern description',
                question_template: 'Question template with {{placeholder}}',
                answer_type: 'multiple',
                difficulty: 'medium',
                description: ''
            }};
            patterns.push(newPattern);
            editingIndex = patterns.length - 1;
            renderPatterns();
        }}

        async function savePatterns() {{
            try {{
                const data = {{
                    target_table: '{self.patterns_data.get("target_table", "wamex_reports") if self.patterns_data else "wamex_reports"}',
                    total_patterns: patterns.length,
                    patterns: patterns
                }};

                const response = await fetch('/api/save', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(data)
                }});

                const result = await response.json();
                if (result.success) {{
                    showStatus('Patterns saved successfully!', 'success');
                }} else {{
                    showStatus('Failed to save: ' + (result.error || 'Unknown error'), 'error');
                }}
            }} catch (error) {{
                showStatus('Failed to save: ' + error.message, 'error');
            }}
        }}

        function showStatus(message, type) {{
            const statusEl = document.getElementById('status');
            statusEl.textContent = message;
            statusEl.className = 'status-message ' + type + ' show';
            setTimeout(() => {{
                statusEl.className = 'status-message ' + type;
            }}, 3000);
        }}

        // Load patterns on page load
        loadPatterns();
    </script>
</body>
</html>"""


def review_patterns_web(
    patterns: List[PathPattern],
    patterns_file: Path,
    port: int = 8080,
) -> List[PathPattern]:
    """Review patterns using web interface.

    Args:
        patterns: List of patterns to review
        patterns_file: Path to save patterns file
        port: Port for web server

    Returns:
        List of reviewed patterns (loaded from file after save)
    """
    import socket

    # Check if port is available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        sock.close()
    except OSError:
        click.echo(f"⚠️  Port {port} is already in use, trying {port + 1}")
        port += 1

    # Create server instance
    server = HTTPServer(("localhost", port), PatternReviewHandler)

    # Load patterns data
    target_table = "wamex_reports"
    if patterns_file.exists():
        with open(patterns_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            target_table = data.get("target_table", target_table)

    PatternReviewHandler.patterns_file = patterns_file
    PatternReviewHandler.patterns_data = {
        "target_table": target_table,
        "total_patterns": len(patterns),
        "patterns": [p.to_dict() for p in patterns],
    }

    url = f"http://localhost:{port}"

    click.echo("\n🌐 Starting pattern review server...")
    click.echo(f"   Server running at: {url}")
    click.echo("   Opening browser...")
    click.echo("\n   Instructions:")
    click.echo("   1. Review and edit patterns in the browser")
    click.echo("   2. Click 'Save Changes' button when done")
    click.echo("   3. Press Enter here to continue")

    # Open browser
    try:
        webbrowser.open(url)
    except Exception as e:
        click.echo(f"   ⚠️  Could not open browser automatically: {e}")
        click.echo(f"   Please open {url} manually")

    # Start server in a thread
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Wait for user to finish
    try:
        click.prompt("\n   Press Enter after you've saved the patterns", default="")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # Shutdown server
        server.shutdown()
        server.server_close()
        click.echo("\n✓ Server stopped")

    # Reload patterns from file
    if patterns_file.exists():
        with open(patterns_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        patterns = [PathPattern.from_dict(p) for p in data["patterns"]]
        click.echo(f"✓ Loaded {len(patterns)} patterns from {patterns_file}")
        return patterns
    else:
        click.echo("⚠️  Patterns file not found, returning original patterns")
        return patterns
