#!/bin/bash
# Deploy Streamlit to Azure Container Apps

set -e

# Configuration
ACR_NAME="btcnadhir2026"
ACR_URL="${ACR_NAME}.azurecr.io"
RESOURCE_GROUP="btc-mlops-prod"
CONTAINER_APP_ENV="env-btc-mlops"
CONTAINER_APP_NAME="btc-streamlit-dashboard"
REGION="switzerlandnorth"
IMAGE_NAME="btc-streamlit-dashboard"
IMAGE_TAG="latest"
FULL_IMAGE="${ACR_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🚀 Deploying Streamlit to Azure Container Apps"
echo "=========================================="
echo "ACR: $ACR_URL"
echo "App: $CONTAINER_APP_NAME"
echo "Region: $REGION"
echo ""

# Build and push image using ACR tasks
echo "📦 Building image in ACR..."
az acr build \
  --registry "$ACR_NAME" \
  --image "${IMAGE_NAME}:${IMAGE_TAG}" \
  --file Dockerfile.streamlit \
  .

echo ""
echo "✅ Image built and pushed to $FULL_IMAGE"
echo ""

# Check if container app exists
echo "🔍 Checking if container app exists..."
if az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  &>/dev/null; then
  
  echo "📝 Updating existing container app..."
  
  az containerapp update \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$FULL_IMAGE"
  
else
  echo "🆕 Creating new container app..."
  
  az containerapp create \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_APP_ENV" \
    --image "$FULL_IMAGE" \
    --target-port 8501 \
    --ingress 'external' \
    --cpu 0.5 \
    --memory 1.0Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars \
      API_URL="https://btc-prediction-api.whitesmoke-94ae13ff.switzerlandnorth.azurecontainerapps.io" \
    --registry-server "$ACR_URL" \
    --registry-username "00000000-0000-0000-0000-000000000000" \
    --registry-password "$(az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv)"

fi

echo ""
echo "✅ Streamlit deployed successfully!"
echo ""

# Get the URL
echo "🌐 Getting container app URL..."
URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv)

echo ""
echo "=========================================="
echo "✨ Streamlit Dashboard is live!"
echo "URL: https://$URL"
echo "=========================================="
