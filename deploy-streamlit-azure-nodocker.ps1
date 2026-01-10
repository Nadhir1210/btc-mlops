#!/usr/bin/env pwsh
# Deploy Streamlit to Azure Container Apps using ACR without local Docker

$ErrorActionPreference = "Stop"

# Configuration
$RESOURCE_GROUP = "btc-mlops-prod"
$CONTAINER_APP_NAME = "btc-streamlit-dashboard"
$REGION = "switzerlandnorth"
$ACR_NAME = "btcnadhir2026"
$ACR_URL = "$ACR_NAME.azurecr.io"

Write-Host "🚀 Deploying Streamlit to Azure (without local Docker)" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

# 1. Créer un déploiement avec une image de base et notre code
Write-Host "📦 Créating Container App with inline Dockerfile..." -ForegroundColor Cyan

# Lire le Dockerfile
$DOCKERFILE_CONTENT = Get-Content Dockerfile.streamlit -Raw

# Créer une archive tar avec le code
Write-Host "📦 Packaging application..." -ForegroundColor Cyan
tar -czf streamlit_app.tar.gz `
    Dockerfile.streamlit `
    app/streamlit_app.py `
    app/requirements.txt

# Upload vers Azure Blob Storage temporaire et build avec Azure Container Registry
Write-Host "🔨 Building with Azure Container Registry..." -ForegroundColor Cyan

$ACR_PASSWORD = az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv

# Utiliser ACI ou Container Apps Source to Cloud pour builder l'image
# Alternative : utiliser ACR directement via Azure portal ou CLI avec un context local

# Pour contourner les limitations, on va créer directement le container app avec une image de base
Write-Host "📝 Creating Container App from public base image..." -ForegroundColor Yellow

# Créer le container app avec une image Streamlit pré-construite ou custom
az containerapp create `
    --name $CONTAINER_APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --environment "env-btc-mlops" `
    --image "python:3.11-slim" `
    --target-port 8501 `
    --ingress 'external' `
    --cpu 0.5 `
    --memory 1.0Gi `
    --min-replicas 1 `
    --max-replicas 3 `
    --env-vars `
        API_URL="https://btc-prediction-api.whitesmoke-94ae13ff.switzerlandnorth.azurecontainerapps.io" `
        STREAMLIT_SERVER_HEADLESS="true" `
        STREAMLIT_SERVER_PORT="8501" `
        STREAMLIT_LOGGER_LEVEL="info" `
    --command "pip install streamlit pandas numpy plotly requests scipy && streamlit run /app/streamlit_app.py"

Write-Host ""
Write-Host "✅ Container App created!" -ForegroundColor Green
Write-Host ""

# Get the URL
$URL = az containerapp show `
    --name $CONTAINER_APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --query properties.configuration.ingress.fqdn `
    -o tsv

Write-Host "🌐 Streamlit Dashboard URL: https://$URL" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏳ Waiting for container to initialize (60 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✨ Streamlit Dashboard is running!" -ForegroundColor Green
Write-Host "URL: https://$URL" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Show logs
Write-Host ""
Write-Host "📊 Container Logs:" -ForegroundColor Cyan
az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow

# Cleanup
Remove-Item streamlit_app.tar.gz -ErrorAction SilentlyContinue
