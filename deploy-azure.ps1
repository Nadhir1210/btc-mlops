az account show
# ============================================================
# Azure Deployment Script for BTC Prediction API (PowerShell)
# Prerequisites: Azure CLI installed and logged in
# ============================================================

$ErrorActionPreference = "Stop"

# Configuration - MODIFY THESE VALUES
$RESOURCE_GROUP = "btc-mlops-rg"
$LOCATION = "westeurope"
$ACR_NAME = "btcmlopsacr"  # Must be globally unique, lowercase, alphanumeric only
$APP_NAME = "btc-prediction-api"
$IMAGE_NAME = "btc-prediction-api"
$IMAGE_TAG = "latest"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Azure Deployment - BTC Prediction API" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Step 1: Login to Azure
Write-Host ""
Write-Host "[1/7] Checking Azure login..." -ForegroundColor Yellow
try {
    az account show | Out-Null
    Write-Host "   Already logged in" -ForegroundColor Green
} catch {
    Write-Host "   Logging in to Azure..." -ForegroundColor Yellow
    az login
}

# Step 2: Create Resource Group
Write-Host ""
Write-Host "[2/7] Creating Resource Group: $RESOURCE_GROUP..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Step 3: Create Azure Container Registry
Write-Host ""
Write-Host "[3/7] Creating Azure Container Registry: $ACR_NAME..." -ForegroundColor Yellow
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true `
    --output table

# Step 4: Get ACR credentials
Write-Host ""
Write-Host "[4/7] Getting ACR credentials..." -ForegroundColor Yellow
$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer --output tsv
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username --output tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv

Write-Host "   ACR Server: $ACR_LOGIN_SERVER" -ForegroundColor Green

# Step 5: Login to ACR
Write-Host ""
Write-Host "[5/7] Logging into ACR..." -ForegroundColor Yellow
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD

# Step 6: Tag and Push image to ACR
Write-Host ""
Write-Host "[6/7] Tagging and pushing image to ACR..." -ForegroundColor Yellow
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

# Step 7: Deploy to Azure Container Instances
Write-Host ""
Write-Host "[7/7] Deploying to Azure Container Instances..." -ForegroundColor Yellow
az container create `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $ACR_USERNAME `
    --registry-password $ACR_PASSWORD `
    --dns-name-label $APP_NAME `
    --ports 8000 `
    --cpu 1 `
    --memory 2 `
    --restart-policy Always `
    --output table

# Get the public URL
$FQDN = az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query ipAddress.fqdn --output tsv

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "   API URL: http://${FQDN}:8000" -ForegroundColor Cyan
Write-Host "   Health:  http://${FQDN}:8000/health" -ForegroundColor Cyan
Write-Host "   Swagger: http://${FQDN}:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "   ACR: $ACR_LOGIN_SERVER" -ForegroundColor White
Write-Host "   Resource Group: $RESOURCE_GROUP" -ForegroundColor White
Write-Host ""
Write-Host "   To view logs:" -ForegroundColor Yellow
Write-Host "   az container logs --resource-group $RESOURCE_GROUP --name $APP_NAME"
Write-Host ""
Write-Host "   To delete all resources:" -ForegroundColor Yellow
Write-Host "   az group delete --name $RESOURCE_GROUP --yes --no-wait"
Write-Host ""
