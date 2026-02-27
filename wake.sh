#!/bin/bash

# ─── CONFIG ────────────────────────────────────────────────────────
PC_MAC="A4:AE:11:F3:9C:22"   # Replace with YOUR PC's Ethernet MAC
PC_IP="192.168.2.1"
MOUNT_POINT="/mnt/pc_inbox"
WAIT_SECONDS=25

# ─── STEP 1: SEND MAGIC PACKET ─────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Sending Magic Packet to $PC_MAC..."
sudo etherwake -i eth0 $PC_MAC

# ─── STEP 2: WAIT FOR WINDOWS TO BOOT ─────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting ${WAIT_SECONDS}s for Windows SMB service..."
sleep $WAIT_SECONDS

# ─── STEP 3: VERIFY PC IS REACHABLE ────────────────────────────────
echo "[$(date '+%H:%M:%S')] Pinging PC to confirm it is online..."
if ping -c 2 -W 3 $PC_IP > /dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] PC is online."
else
    echo "[$(date '+%H:%M:%S')] WARNING: PC not responding to ping."
fi

# ─── STEP 4: FORCE MOUNT REFRESH ────────────────────────────────────
sudo umount $MOUNT_POINT 2>/dev/null
sudo mount $MOUNT_POINT

# ─── STEP 5: VERIFY MOUNT ────────────────────────────────────────────
if mountpoint -q $MOUNT_POINT; then
    echo "[$(date '+%H:%M:%S')] SUCCESS: $MOUNT_POINT is mounted and ready."
else
    echo "[$(date '+%H:%M:%S')] ERROR: Mount failed."
fi
