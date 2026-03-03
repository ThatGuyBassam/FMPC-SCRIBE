import os
import time
import shutil
import subprocess
import logging

# ─── CONFIG ────────────────────────────────────────────────────────
# Dynamically gets the home directory (e.g., /home/bassam) 
HOME_DIR  = os.path.expanduser("~") 
STAGING   = os.path.join(HOME_DIR, "fmpc_staging")
LOG_PATH  = os.path.join(HOME_DIR, "pi_watchdog.log")

# These match your v3.0 Manual specifications
INBOX     = "/mnt/pc_inbox"
PC_IP     = "192.168.2.1"
WAIT_BOOT = 25  # Seconds to wait for Windows to POST/Boot
SUPPORTED = ('.m4a', '.mp3', '.wav', '.aac', '.ogg', '.flac')

# PLACEHOLDERS: Replace with your actual values ONLY on the Pi hardware
PC_MAC    = "A4:AE:11:F3:9C:22" 

# ─── LOGGING ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

os.makedirs(STAGING, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────
def check_bridge_health():
    """
    Harden the connection: Checks if the Ethernet bridge is physically active.
    Attempts to restart the connection if the cable was unplugged/replugged.
    """
    try:
        # Check the physical state of the direct Ethernet link
        status = os.popen("cat /sys/class/net/eth0/operstate").read().strip()
        if status != "up":
            log.warning("Bridge interface eth0 is DOWN. Attempting restart...")
            subprocess.run(["sudo", "nmcli", "connection", "up", "Wired connection 1"], capture_output=True)
        return status == "up"
    except Exception as e:
        log.error(f"Bridge health check failed: {e}")
        return False

def send_wol():
    """Sends the Magic Packet via the dedicated eth0 bridge interface."""
    log.info(f"Sending Magic Packet to {PC_MAC}...")
    subprocess.run(["sudo", "etherwake", "-i", "eth0", PC_MAC])

def wait_for_pc():
    """Pings the PC to confirm Windows is online before moving files."""
    log.info(f"Waiting {WAIT_BOOT}s for Windows to boot...")
    time.sleep(WAIT_BOOT)
    result = subprocess.run(
        ["ping", "-c", "2", "-W", "3", PC_IP],
        capture_output=True
    )
    if result.returncode == 0:
        log.info("PC is online.")
        return True
    else:
        log.warning("PC not responding to ping. The engine will still attempt to mount.")
        return False

def refresh_mount():
    """Forces a refresh of the SMB share to ensure the file transfer doesn't hang."""
    subprocess.run(["sudo", "umount", INBOX], capture_output=True)
    result = subprocess.run(["sudo", "mount", INBOX], capture_output=True)
    if result.returncode == 0:
        log.info("SMB mount refreshed successfully.")
        return True
    else:
        log.error("Mount failed. Check credentials in /etc/samba/fmpc_credentials.")
        return False

def is_file_complete(path, wait=3):
    """Verify upload is finished by checking if the file size is stable."""
    try:
        size1 = os.path.getsize(path)
        time.sleep(wait)
        size2 = os.path.getsize(path)
        return size1 == size2
    except OSError:
        return False

# ─── WATCHDOG LOOP ─────────────────────────────────────────────────
[Image of a network heartbeat signal diagram]
log.info("=" * 50)
log.info("FMPC PI WATCHDOG v3.1 — ACTIVE")
log.info(f"Watching: {STAGING}")
log.info(f"Target: {PC_IP} ({PC_MAC})")
log.info("=" * 50)

while True:
    try:
        # Scan for supported medical lecture formats
        files = [f for f in os.listdir(STAGING) if f.lower().endswith(SUPPORTED)]

        if files:
            log.info(f"Detected {len(files)} file(s) in staging.")

            # 1. Ensure bridge is physically connected
            check_bridge_health()

            # 2. Wake the RTX 4060 "Brain"
            send_wol()
            wait_for_pc()

            # 3. Connect to the PC's INBOX share
            if not refresh_mount():
                log.error("Aborting transfer: Mount unavailable. Retrying in 30s.")
                time.sleep(30)
                continue

            # 4. Transfer files
            for fname in files:
                src = os.path.join(STAGING, fname)

                # Ensure Solid Explorer has finished the SFTP upload
                log.info(f"Verifying upload integrity: {fname}")
                while not is_file_complete(src):
                    time.sleep(2)

                dst = os.path.join(INBOX, fname)
                shutil.move(src, dst)
                log.info(f"Successfully moved to PC INBOX: {fname}")

            log.info("Sequence complete. Returning to standby.")

    except Exception as e:
        log.error(f"Watchdog Loop Error: {e}")

    time.sleep(10) # Poll every 10 seconds to save Pi CPU cycles
