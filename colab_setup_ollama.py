#!/usr/bin/env python3
"""
colab_setup_ollama.py
---------------------
Automatiza, no Google Colab:

  1. Instalação do Ollama (script oficial).
  2. Subida do servidor Ollama em 0.0.0.0:<porta>.
  3. Download do modelo (padrão: qwen3:14b).
  4. Exposição pública via túnel SSH do Pinggy (versão GRATUITA, sem token).
  5. Impressão dos endpoints (nativo e compatível com OpenAI).

Baseado nas respostas do projeto:
  - Túnel : pinggy (free, sem token)
  - Modelo: qwen3:14b
  - Auth  : nenhuma
  - Extras: nenhum

Uso típico em uma célula do Colab:

    !python colab_setup_ollama.py

    !python colab_setup_ollama.py --model qwen3:14b --port 11434

    !python colab_setup_ollama.py --tunnel none       # só local

    !python colab_setup_ollama.py --stop              # derruba tudo

Veja todas as opções com:

    !python colab_setup_ollama.py --help
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import NoReturn

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"

PINGGY_HOST = "a.pinggy.io"
PINGGY_SSH_PORT = 443
PINGGY_DEFAULT_USER = "free"
PINGGY_URL_RE = re.compile(
    r"https://[a-zA-Z0-9.\-]+\.(?:pinggy\.io|pinggy\.link|pinggy\.online)"
)

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_PORT = 11434
DEFAULT_STATE_DIR = Path("/content/ollama-colab")

# Assinatura esperada na linha de comando de cada processo gerenciado.
# Usada para não encerrar PIDs reaproveitados por outros processos.
PROCESS_SIGNATURES = {
    "ollama": "ollama",
    "pinggy": "ssh",
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[setup-ollama] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[setup-ollama][aviso] {msg}", flush=True)


def fail(msg: str, code: int = 1) -> NoReturn:
    print(f"[setup-ollama][erro] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def run(cmd: list[str], env: dict | None = None) -> None:
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def ensure_state_dir(state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def write_pid(state_dir: Path, name: str, pid: int) -> None:
    (state_dir / f"{name}.pid").write_text(str(pid))


def read_pid(state_dir: Path, name: str) -> int | None:
    path = state_dir / f"{name}.pid"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _read_cmdline(pid: int) -> str:
    """Lê /proc/<pid>/cmdline. Retorna '' se o processo não existir mais."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode(errors="ignore").strip()


def kill_pid(state_dir: Path, name: str) -> None:
    pid = read_pid(state_dir, name)
    if pid is None:
        return

    cmdline = _read_cmdline(pid)
    if not cmdline:
        # PID já não existe (ou /proc indisponível): apenas limpa o registro.
        log(f"{name}: PID {pid} não está mais ativo; removendo PID obsoleto.")
        (state_dir / f"{name}.pid").unlink(missing_ok=True)
        return

    expected = PROCESS_SIGNATURES.get(name)
    if expected and expected not in cmdline:
        warn(
            f"{name}: PID {pid} foi reaproveitado por outro processo "
            f"('{cmdline[:80]}'); não vou encerrá-lo. Removendo PID obsoleto."
        )
        (state_dir / f"{name}.pid").unlink(missing_ok=True)
        return

    try:
        os.kill(pid, 15)  # SIGTERM
        log(f"{name}: processo {pid} encerrado (SIGTERM).")
    except ProcessLookupError:
        log(f"{name}: processo {pid} já não existia.")
    except PermissionError:
        warn(f"{name}: sem permissão para encerrar o PID {pid}.")
    finally:
        (state_dir / f"{name}.pid").unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def report_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        warn("nvidia-smi não encontrado — rodando provavelmente em CPU (será lento).")
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        log(f"GPU detectada: {out}")
    except subprocess.CalledProcessError:
        warn("Falha ao consultar a GPU via nvidia-smi.")


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _find_ollama() -> str | None:
    """Procura o binário do Ollama no PATH e em locais comuns do Colab."""
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in ("/usr/local/bin/ollama", "/usr/bin/ollama"):
        if Path(candidate).exists():
            return candidate
    return None


def ollama_bin() -> str:
    found = _find_ollama()
    if found:
        return found
    fail("Binário 'ollama' não encontrado após a instalação.")


def install_ollama() -> None:
    if _find_ollama():
        log("Ollama já está instalado.")
        return

    installer = Path("/tmp/ollama_install.sh")
    log("Baixando o instalador oficial do Ollama...")
    subprocess.run(
        ["curl", "-fsSL", OLLAMA_INSTALL_URL, "-o", str(installer)], check=True
    )
    log("Executando o instalador (pode levar 1-2 min)...")
    # O instalador tenta iniciar via systemd; no Colab isso falha de forma
    # inofensiva, então não usamos check=True.
    subprocess.run(["bash", str(installer)])
    log("Instalação concluída.")


def ollama_env(port: int) -> dict:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    env["OLLAMA_ORIGINS"] = "*"
    env["OLLAMA_KEEP_ALIVE"] = "-1"  # mantém o modelo carregado na VRAM
    return env


def start_ollama(port: int, state_dir: Path) -> None:
    local_url = f"http://127.0.0.1:{port}/api/tags"
    if http_ok(local_url):
        log(f"Servidor Ollama já está respondendo em 127.0.0.1:{port}.")
        return

    logfile = open(state_dir / "ollama.log", "ab")
    proc = subprocess.Popen(
        [ollama_bin(), "serve"],
        env=ollama_env(port),
        stdin=subprocess.DEVNULL,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # sobrevive ao fim da célula
    )
    write_pid(state_dir, "ollama", proc.pid)
    log(f"Servidor iniciado (PID {proc.pid}), log em {state_dir / 'ollama.log'}")


def wait_for_ollama(port: int, timeout: int = 90) -> None:
    local_url = f"http://127.0.0.1:{port}/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_ok(local_url):
            log("Servidor Ollama pronto.")
            return
        time.sleep(2)
    fail(f"Ollama não respondeu em {timeout}s — veja ollama.log.")


def model_present(model: str, port: int) -> bool:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
    try:
        out = subprocess.check_output(
            [ollama_bin(), "list"], env=env, text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return False
    return any(
        line.split()[0] == model for line in out.splitlines()[1:] if line.strip()
    )


def pull_model(model: str, port: int) -> None:
    if model_present(model, port):
        log(f"Modelo '{model}' já está baixado.")
        return
    log(f"Baixando o modelo '{model}' (pode levar vários minutos)...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
    run([ollama_bin(), "pull", model], env=env)
    log(f"Modelo '{model}' disponível.")


def smoke_test(model: str, port: int) -> None:
    log("Executando um teste rápido de inferência...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
    try:
        out = subprocess.check_output(
            [ollama_bin(), "run", model, "Responda apenas com: ok"],
            env=env,
            text=True,
            timeout=300,
        )
        log(f"Resposta do modelo: {out.strip()[:200]}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        warn(
            f"Teste de inferência falhou ({exc.__class__.__name__}). "
            "O modelo ainda pode estar carregando."
        )


# --------------------------------------------------------------------------- #
# Túnel: Pinggy (SSH) — versão gratuita, sem token
# --------------------------------------------------------------------------- #
def pinggy_ssh_command(port: int, user: str) -> list[str]:
    return [
        "ssh",
        "-p",
        str(PINGGY_SSH_PORT),
        "-R0:localhost:" + str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        "-T",
        f"{user}@{PINGGY_HOST}",
    ]


def start_pinggy(
    port: int,
    user: str,
    state_dir: Path,
    timeout: int = 90,
) -> str:
    if shutil.which("ssh") is None:
        fail("Cliente 'ssh' não encontrado no ambiente (inesperado no Colab).")

    ssh_user = (user or PINGGY_DEFAULT_USER).strip()
    warn(
        "Usando o túnel GRATUITO do Pinggy (sem token): URL aleatória, limites "
        "de banda/duração do plano free e uma página de aviso é injetada em "
        "requisições de navegador. Clientes HTTP comuns (curl/SDKs) não sofrem "
        "com essa página."
    )

    if read_pid(state_dir, "pinggy") is not None:
        warn("Já existe um túnel pinggy registrado; encerrando antes de subir outro.")
        kill_pid(state_dir, "pinggy")
        time.sleep(2)

    logfile_path = state_dir / "pinggy.log"
    # truncamos o log para não pegar URL antiga
    logfile_path.write_text("")
    logfile = open(logfile_path, "ab")

    cmd = pinggy_ssh_command(port, ssh_user)
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,  # nunca travar esperando senha
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    write_pid(state_dir, "pinggy", proc.pid)
    log(f"Túnel pinggy iniciado (PID {proc.pid}), log em {logfile_path}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        text = logfile_path.read_text(errors="ignore")
        match = PINGGY_URL_RE.search(text)
        if match:
            return match.group(0)
        if proc.poll() is not None:
            fail(
                f"O SSH do pinggy encerrou (código {proc.returncode}). "
                f"Veja {logfile_path}."
            )
        time.sleep(2)

    fail(f"Não foi possível obter a URL do pinggy em {timeout}s. Veja {logfile_path}.")


def verify_tunnel(public_url: str, timeout: float = 20.0) -> bool:
    """Confirma que o túnel está roteando para o Ollama."""
    url = public_url.rstrip("/") + "/api/tags"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 400:
                    return True
        except Exception:
            time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# Encerrar tudo
# --------------------------------------------------------------------------- #
def stop_all(state_dir: Path) -> None:
    log("Encerrando processos...")
    kill_pid(state_dir, "pinggy")
    kill_pid(state_dir, "ollama")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Instala e configura Ollama + qwen3:14b + túnel Pinggy (free) no "
            "Google Colab."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"modelo a baixar (padrão: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"porta local do Ollama (padrão: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--tunnel",
        choices=("pinggy", "none"),
        default="pinggy",
        help="tipo de túnel público (padrão: pinggy)",
    )
    parser.add_argument(
        "--pinggy-user",
        default=PINGGY_DEFAULT_USER,
        help=(
            "usuário SSH do Pinggy (padrão: free). O plano gratuito não usa "
            "token; se um dia você tiver token, ele pode ir aqui."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE_DIR),
        help="diretório de logs/PIDs",
    )
    parser.add_argument(
        "--skip-pull", action="store_true", help="não baixar o modelo"
    )
    parser.add_argument(
        "--no-test", action="store_true", help="não rodar o teste de inferência"
    )
    parser.add_argument(
        "--stop", action="store_true", help="encerra Ollama e túnel e sai"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_dir = ensure_state_dir(Path(args.state_dir))

    if args.stop:
        stop_all(state_dir)
        return 0

    report_gpu()

    install_ollama()
    start_ollama(args.port, state_dir)
    wait_for_ollama(args.port)

    if not args.skip_pull:
        pull_model(args.model, args.port)

    if not args.no_test and not args.skip_pull:
        smoke_test(args.model, args.port)

    public_url = None
    tunnel_ok = False
    if args.tunnel == "pinggy":
        public_url = start_pinggy(
            args.port,
            args.pinggy_user,
            state_dir,
        )
        tunnel_ok = verify_tunnel(public_url)
        if tunnel_ok:
            log("Túnel pinggy validado (GET /api/tags respondeu).")
        else:
            warn(
                f"O túnel respondeu pela URL {public_url}, mas o teste "
                "GET /api/tags falhou. Verifique pinggy.log."
            )

    local_base = f"http://127.0.0.1:{args.port}"
    print()
    print("=" * 68)
    print(" Ollama pronto")
    print("=" * 68)
    print(f" Modelo          : {args.model}")
    print(f" API local       : {local_base}")
    print(f" API OpenAI-like : {local_base}/v1")
    if public_url:
        print(f" Túnel público   : {public_url}  (Pinggy free)")
        print(f" API OpenAI-like : {public_url}/v1")
        print()
        print(" Exemplo (de fora do Colab):")
        print(f"   curl {public_url}/v1/chat/completions \\")
        print('     -H "Content-Type: application/json" \\')
        print(
            f'     -d \'{{"model": "{args.model}", "messages": '
            '[{"role": "user", "content": "olá"}]}\''
        )
        print()
        warn("O túnel público NÃO tem autenticação. Derrube com: --stop")
        if not tunnel_ok:
            warn("A validação automática do túnel falhou; teste manualmente.")
    else:
        print(" Túnel           : desativado")
    print("=" * 68)
    print(f" Logs e PIDs em  : {state_dir}")
    print(f" Para encerrar   : !python {Path(__file__).name} --stop")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        log("Interrompido pelo usuário.")
        raise SystemExit(130)
