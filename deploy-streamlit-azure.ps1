# Deploy Streamlit to Azure Container Apps (PowerShell)

$ErrorActionPreference = "Stop"

# Configuration
$ACR_NAME = "btcnadhir2026"
$ACR_URL = "$ACR_NAME.azurecr.io"
$RESOURCE_GROUP = "btc-mlops-prod"
$CONTAINER_APP_ENV = "env-btc-mlops"
$CONTAINER_APP_NAME = "btc-streamlit-dashboard"
$REGION = "switzerlandnorth"
$IMAGE_NAME = "btc-streamlit-dashboard"
$IMAGE_TAG = "latest"
$FULL_IMAGE = "$ACR_URL/$IMAGE_NAME`:$IMAGE_TAG"

Write-Host "🚀 Deploying Streamlit to Azure Container Apps" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "ACR: $ACR_URL"
Write-Host "App: $CONTAINER_APP_NAME"
Write-Host "Region: $REGION"
Write-Host ""

# Build and push image using ACR tasks
Write-Host "📦 Building image in ACR..." -ForegroundColor Cyan
az acr build `
  --registry $ACR_NAME `
  --image "$IMAGE_NAME`:$IMAGE_TAG" `
  --file Dockerfile.streamlit `
  .

Write-Host ""
Write-Host "✅ Image built and pushed to $FULL_IMAGE" -ForegroundColor Green
Write-Host ""

# Check if container app exists
Write-Host "🔍 Checking if container app exists..." -ForegroundColor Cyan
$appExists = az containerapp show `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  2>$null

if ($appExists) {
  
  Write-Host "📝 Updating existing container app..." -ForegroundColor Yellow
  
  az containerapp update `
    --name $CONTAINER_APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --image $FULL_IMAGE
  
} else {
  Write-Host "🆕 Creating new container app..." -ForegroundColor Yellow
  
  # Get ACR password
  $ACR_PASSWORD = az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv
  
  az containerapp create `
    --name $CONTAINER_APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --environment $CONTAINER_APP_ENV `
    --image $FULL_IMAGE `
    --target-port 8501 `
    --ingress 'external' `
    --cpu 0.5 `
    --memory 1.0Gi `
    --min-replicas 1 `
    --max-replicas 3 `
    --env-vars API_URL="https://btc-prediction-api.whitesmoke-94ae13ff.switzerlandnorth.azurecontainerapps.io" `
    --registry-server $ACR_URL `
    --registry-username "00000000-0000-0000-0000-000000000000" `
    --registry-password $ACR_PASSWORD
}

Write-Host ""
Write-Host "✅ Streamlit deployed successfully!" -ForegroundColor Green
Write-Host ""

# Get the URL
Write-Host "🌐 Getting container app URL..." -ForegroundColor Cyan
$URL = az containerapp show `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query properties.configuration.ingress.fqdn `
  -o tsv

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✨ Streamlit Dashboard is live!" -ForegroundColor Green
Write-Host "URL: https://$URL" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

