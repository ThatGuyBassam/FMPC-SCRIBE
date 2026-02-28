import os, time, shutil, subprocess, logging

# ─── CONFIG ────────────────────────────────────────────────────────
# will replace YOUR_USERNAME with my Pi username when code is in use
# will replace the MAC address with my PC's Ethernet MAC (from getmac /v on Windows)
STAGING   = "/home/YOUR_USERNAME/fmpc_staging"
INBOX     = "/mnt/pc_inbox"
PC_MAC    = "A4:AE:11:F3:9C:22"
PC_IP     = "192.168.2.1"
WAIT_BOOT = 25   # seconds to wait for Windows to boot after Magic Packet
LOG_PATH  = "/home/YOUR_USERNAME/pi_watchdog.log"

SUPPORTED = ('.m4a', '.mp3', '.wav', '.aac', '.ogg', '.flac')

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
def send_wol():
    log.info(f"Sending Magic Packet to {PC_MAC}...")
    subprocess.run(["sudo", "etherwake", "-i", "eth0", PC_MAC])

def wait_for_pc():
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
        log.warning("PC not responding to ping. Proceeding anyway...")
        return False

def refresh_mount():
    subprocess.run(["sudo", "umount", INBOX], capture_output=True)
    result = subprocess.run(["sudo", "mount", INBOX], capture_output=True)
    if result.returncode == 0:
        log.info("SMB mount refreshed successfully.")
        return True
    else:
        log.error("Mount failed. Check credentials and PC share settings.")
        return False

def is_file_complete(path, wait=3):
    """
    Verify upload is finished by checking file size is stable.
    Prevents processing a partially uploaded file.
    """
    size1 = os.path.getsize(path)
    time.sleep(wait)
    size2 = os.path.getsize(path)
    return size1 == size2

# ─── WATCHDOG LOOP ─────────────────────────────────────────────────
log.info("=" * 50)
log.info("PI WATCHDOG — STARTED")
log.info(f"Watching: {STAGING}")
log.info(f"PC MAC: {PC_MAC} | PC IP: {PC_IP}")
log.info("=" * 50)

while True:
    try:
        files = [f for f in os.listdir(STAGING) if f.lower().endswith(SUPPORTED)]

        if files:
            log.info(f"Detected {len(files)} file(s) in staging.")

            # Step 1: Wake the PC
            send_wol()
            wait_for_pc()

            # Step 2: Refresh SMB mount
            if not refresh_mount():
                log.error("Skipping transfer — mount unavailable. Will retry in 30 seconds.")
                time.sleep(30)
                continue

            # Step 3: Move each file to PC INBOX
            for fname in files:
                src = os.path.join(STAGING, fname)

                # Wait until upload is fully complete (size-stable check)
                log.info(f"Waiting for upload to finish: {fname}")
                for _ in range(10):
                    if is_file_complete(src):
                        break
                    time.sleep(3)

                dst = os.path.join(INBOX, fname)
                shutil.move(src, dst)
                log.info(f"Moved to INBOX: {fname}")

            log.info("All files transferred. PC engine will handle the rest.")

    except Exception as e:
        log.error(f"Watchdog error: {e}")

    time.sleep(10)
