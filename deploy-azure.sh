#!/bin/bash
# ============================================================
# Azure Deployment Script for BTC Prediction API
# Prerequisites: Azure CLI installed and logged in
# ============================================================

set -e

# Configuration - MODIFY THESE VALUES
RESOURCE_GROUP="btc-mlops-rg"
LOCATION="westeurope"
ACR_NAME="btcmlopsacr"  # Must be globally unique, lowercase, alphanumeric only
APP_NAME="btc-prediction-api"
IMAGE_NAME="btc-prediction-api"
IMAGE_TAG="latest"

echo "============================================================"
echo "   Azure Deployment - BTC Prediction API"
echo "============================================================"

# Step 1: Login to Azure (if not already logged in)
echo ""
echo "[1/7] Checking Azure login..."
az account show > /dev/null 2>&1 || az login

# Step 2: Create Resource Group
echo ""
echo "[2/7] Creating Resource Group: $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Step 3: Create Azure Container Registry
echo ""
echo "[3/7] Creating Azure Container Registry: $ACR_NAME..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output table

# Step 4: Get ACR credentials
echo ""
echo "[4/7] Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

echo "   ACR Server: $ACR_LOGIN_SERVER"

# Step 5: Login to ACR
echo ""
echo "[5/7] Logging into ACR..."
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD

# Step 6: Tag and Push image to ACR
echo ""
echo "[6/7] Tagging and pushing image to ACR..."
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG

# Step 7: Deploy to Azure Container Instances
echo ""
echo "[7/7] Deploying to Azure Container Instances..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label $APP_NAME \
    --ports 8000 \
    --cpu 1 \
    --memory 2 \
    --restart-policy Always \
    --output table

# Get the public URL
FQDN=$(az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query ipAddress.fqdn --output tsv)

echo ""
echo "============================================================"
echo "   DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "   API URL: http://$FQDN:8000"
echo "   Health:  http://$FQDN:8000/health"
echo "   Swagger: http://$FQDN:8000/docs"
echo ""
echo "   ACR: $ACR_LOGIN_SERVER"
echo "   Resource Group: $RESOURCE_GROUP"
echo ""
echo "   To view logs:"
echo "   az container logs --resource-group $RESOURCE_GROUP --name $APP_NAME"
echo ""
echo "   To delete all resources:"
echo "   az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""
