#!/bin/bash
# MedBharat-1 OCI Deployment Script
#
# Prerequisites:
#   1. OCI CLI configured (oci setup config)
#   2. Docker installed
#   3. OCIR (OCI Container Registry) access
#   4. Trained model weights in ./weights/
#
# Usage:
#   ./deploy/oci_deploy.sh <region> <tenancy-namespace> <compartment-ocid>
#
# Example:
#   ./deploy/oci_deploy.sh ap-mumbai-1 mytenancy ocid1.compartment.oc1..xxx

set -euo pipefail

# --- Arguments ---
REGION="${1:?Usage: $0 <region> <tenancy-namespace> <compartment-ocid>}"
NAMESPACE="${2:?Provide tenancy namespace}"
COMPARTMENT_OCID="${3:?Provide compartment OCID}"

IMAGE_NAME="medbharat-1"
IMAGE_TAG="latest"
REGISTRY="${REGION}.ocir.io"
FULL_IMAGE="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================"
echo "  MedBharat-1 OCI Deployment"
echo "============================================"
echo "Region:    ${REGION}"
echo "Registry:  ${FULL_IMAGE}"
echo ""

# --- Step 1: Build Docker image ---
echo "[1/5] Building Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f deploy/Dockerfile .

# --- Step 2: Tag for OCI Registry ---
echo "[2/5] Tagging image for OCIR..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"

# --- Step 3: Login to OCI Registry ---
echo "[3/5] Logging in to OCIR..."
echo "Enter your OCI auth token when prompted:"
docker login "${REGISTRY}" -u "${NAMESPACE}/oracleidentitycloudservice/$(oci iam user list --query 'data[0].name' --raw-output 2>/dev/null || echo 'your-email')"

# --- Step 4: Push to OCIR ---
echo "[4/5] Pushing image to OCIR..."
docker push "${FULL_IMAGE}"

# --- Step 5: Create OCI Container Instance ---
echo "[5/5] Creating OCI Container Instance..."

# Check if model weights exist in Object Storage
BUCKET_NAME="medbharat-weights"

echo ""
echo "============================================"
echo "  Next Steps (Manual in OCI Console):"
echo "============================================"
echo ""
echo "1. Upload model weights to OCI Object Storage:"
echo "   oci os bucket create --name ${BUCKET_NAME} --compartment-id ${COMPARTMENT_OCID}"
echo "   oci os object put --bucket-name ${BUCKET_NAME} --file weights/model.pt --name model.pt"
echo "   oci os object put --bucket-name ${BUCKET_NAME} --file weights/medbharat_tokenizer.json --name medbharat_tokenizer.json"
echo ""
echo "2. Create Container Instance (OCI Console or CLI):"
echo "   - Image: ${FULL_IMAGE}"
echo "   - Shape: VM.GPU.A10.1 (or VM.Standard.A1.Flex for CPU)"
echo "   - Memory: 16 GB minimum"
echo "   - Port: 8000"
echo "   - Environment variables:"
echo "     MEDBHARAT_WEIGHTS=/app/weights"
echo "     MEDBHARAT_DEVICE=cuda"
echo ""
echo "3. Create API Gateway route to container port 8000"
echo ""
echo "4. Test:"
echo '   curl -X POST https://<your-api-gateway>/medical_qa \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '\''{"question": "What is the first-line treatment for dengue fever?"}'\'''
echo ""
echo "Done! Image pushed: ${FULL_IMAGE}"
