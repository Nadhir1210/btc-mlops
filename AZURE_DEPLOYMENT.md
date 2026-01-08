# Deploiement Azure avec ACR - Guide Complet

## Prerequisites

1. **Azure CLI** : [Installer Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli-windows)
   ```powershell
   winget install Microsoft.AzureCLI
   ```

2. **Docker** : Deja installe et fonctionnel

3. **Compte Azure** : Avec un abonnement actif

---

## Etape 1 : Installation Azure CLI

```powershell
# Via winget (recommande)
winget install Microsoft.AzureCLI

# OU telecharger depuis
# https://aka.ms/installazurecliwindows
```

Redemarrer le terminal apres l'installation.

---

## Etape 2 : Connexion a Azure

```powershell
# Login interactif
az login

# Verifier la connexion
az account show --output table

# Lister les abonnements
az account list --output table

# Selectionner un abonnement (si plusieurs)
az account set --subscription "NOM_OU_ID_ABONNEMENT"
```

---

## Etape 3 : Deploiement Manuel

### Variables de configuration
```powershell
$RESOURCE_GROUP = "btc-mlops-rg"
$LOCATION = "westeurope"
$ACR_NAME = "btcmlopsacr123"  # DOIT ETRE UNIQUE GLOBALEMENT
$APP_NAME = "btc-prediction-api"
```

### Creer le Resource Group
```powershell
az group create --name $RESOURCE_GROUP --location $LOCATION
```

### Creer Azure Container Registry
```powershell
az acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true
```

### Obtenir les credentials ACR
```powershell
$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer --output tsv
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username --output tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv

Write-Host "Server: $ACR_LOGIN_SERVER"
Write-Host "Username: $ACR_USERNAME"
```

### Login Docker vers ACR
```powershell
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD
```

### Tag et Push de l'image
```powershell
# Tag l'image
docker tag btc-prediction-api:latest "${ACR_LOGIN_SERVER}/btc-prediction-api:latest"

# Push vers ACR
docker push "${ACR_LOGIN_SERVER}/btc-prediction-api:latest"
```

### Deployer sur Azure Container Instances
```powershell
az container create `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --image "${ACR_LOGIN_SERVER}/btc-prediction-api:latest" `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $ACR_USERNAME `
    --registry-password $ACR_PASSWORD `
    --dns-name-label $APP_NAME `
    --ports 8000 `
    --cpu 1 `
    --memory 2 `
    --restart-policy Always
```

### Obtenir l'URL publique
```powershell
$FQDN = az container show `
    --resource-group $RESOURCE_GROUP `
    --name $APP_NAME `
    --query ipAddress.fqdn `
    --output tsv

Write-Host "API URL: http://${FQDN}:8000"
Write-Host "Swagger: http://${FQDN}:8000/docs"
```

---

## Etape 4 : Verification

```powershell
# Tester l'API
curl "http://${FQDN}:8000/health"

# Voir les logs
az container logs --resource-group $RESOURCE_GROUP --name $APP_NAME

# Statut du container
az container show --resource-group $RESOURCE_GROUP --name $APP_NAME --query instanceView.state
```

---

## Commandes Utiles

```powershell
# Redemarrer le container
az container restart --resource-group $RESOURCE_GROUP --name $APP_NAME

# Arreter le container
az container stop --resource-group $RESOURCE_GROUP --name $APP_NAME

# Supprimer le container
az container delete --resource-group $RESOURCE_GROUP --name $APP_NAME --yes

# Supprimer toutes les ressources
az group delete --name $RESOURCE_GROUP --yes --no-wait

# Lister les images dans ACR
az acr repository list --name $ACR_NAME --output table

# Voir les tags d'une image
az acr repository show-tags --name $ACR_NAME --repository btc-prediction-api
```

---

## Couts Estimes

| Service | Configuration | Cout/mois (approx) |
|---------|--------------|-------------------|
| ACR Basic | 10 GB inclus | ~5 EUR |
| ACI | 1 vCPU, 2 GB RAM | ~30-40 EUR |
| **Total** | | **~35-45 EUR/mois** |

---

## Alternative : Azure App Service

Pour un deploiement plus robuste avec auto-scaling :

```powershell
# Creer un App Service Plan
az appservice plan create `
    --name btc-mlops-plan `
    --resource-group $RESOURCE_GROUP `
    --sku B1 `
    --is-linux

# Creer le Web App
az webapp create `
    --resource-group $RESOURCE_GROUP `
    --plan btc-mlops-plan `
    --name btc-prediction-webapp `
    --deployment-container-image-name "${ACR_LOGIN_SERVER}/btc-prediction-api:latest"

# Configurer le port
az webapp config appsettings set `
    --resource-group $RESOURCE_GROUP `
    --name btc-prediction-webapp `
    --settings WEBSITES_PORT=8000
```

---

## CI/CD avec GitHub Actions

Le fichier `.github/workflows/azure-deploy.yml` est configure pour :
1. Build l'image Docker
2. Push vers ACR
3. Deployer sur ACI

### Secrets GitHub requis :
- `AZURE_CREDENTIALS` : JSON de service principal
- `ACR_USERNAME` : Nom d'utilisateur ACR
- `ACR_PASSWORD` : Mot de passe ACR

### Creer les credentials :
```powershell
az ad sp create-for-rbac `
    --name "btc-mlops-sp" `
    --role contributor `
    --scopes /subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP `
    --sdk-auth
```

Copier le JSON genere dans le secret `AZURE_CREDENTIALS`.
