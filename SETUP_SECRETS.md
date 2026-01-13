# 🔐 Configuration des Secrets GitHub - Guide Complet

## 📋 Prérequis

- Azure CLI installé et configuré
- GitHub CLI (gh) installé
- Accès au repository GitHub

---

## Étape 1: Récupérer les Credentials Azure

### 1.1 ACR Credentials

```powershell
# Récupérer les credentials ACR
az acr credential show --name btcmlopsacr
```

**Notez ces valeurs:**
- `username`: btcmlopsacr
- `passwords[0].value`: (le mot de passe)

### 1.2 Service Principal (Déjà créé)

```json
{
  "clientId": "<YOUR_CLIENT_ID>",
  "clientSecret": "<YOUR_CLIENT_SECRET>",
  "subscriptionId": "<YOUR_SUBSCRIPTION_ID>",
  "tenantId": "<YOUR_TENANT_ID>"
}
```

**Note**: Utilisez les valeurs du service principal créé précédemment.

---

## Étape 2: Créer les Secrets GitHub

### Option A: Interface Web GitHub

1. Allez sur: **https://github.com/Nadhir1210/btc-mlops/settings/secrets/actions**

2. Cliquez sur **"New repository secret"**

3. Créez ces 5 secrets:

#### Secret 1: `AZURE_CREDENTIALS_V2`
```json
{
  "clientId": "<YOUR_CLIENT_ID>",
  "clientSecret": "<YOUR_CLIENT_SECRET>",
  "subscriptionId": "<YOUR_SUBSCRIPTION_ID>",
  "tenantId": "<YOUR_TENANT_ID>"
}
```

**Remplacez avec vos valeurs du service principal Azure.**

#### Secret 2: `ACR_USERNAME`
```
btcmlopsacr
```

<YOUR_ACR_PASSWORD>
```

**Récupérez avec**: `az acr credential show --name btcmlopsacr --query "passwords[0].value" -o tsv`
(Coller la valeur de passwords[0].value de la commande az acr credential)
```

#### Secret 4: `ACR_LOGIN_SERVER`
```
btcmlopsacr.azurecr.io
```

#### Secret 5: `AZURE_WEBAPP_NAME`
```
btc-mlops-api
```

### Option B: GitHub CLI (Ligne de commande)

```powershell
# 1. Login GitHub CLI
gh auth login

# 2. Récupérer le password ACR
$acrPassword = (az acr credential show --name btcmlopsacr --query "passwords[0].value" -o tsv)

# 3. Créer les <YOUR_CLIENT_ID>",
  "clientSecret": "<YOUR_CLIENT_SECRET>",
  "subscriptionId": "<YOUR_SUBSCRIPTION_ID>",
  "tenantId": "<YOUR_TENANT_ID>l2oZ9aGl",
  "subscriptionId": "44be7a71-c968-4817-a5da-a30e18af4c15",
  "tenantId": "b7bd4715-4217-48c7-919e-2ea97f592fa7"
}' --repo Nadhir1210/btc-mlops

gh secret set ACR_USERNAME --body "btcmlopsacr" --repo Nadhir1210/btc-mlops

gh secret set ACR_PASSWORD --body $acrPassword --repo Nadhir1210/btc-mlops

gh secret set ACR_LOGIN_SERVER --body "btcmlopsacr.azurecr.io" --repo Nadhir1210/btc-mlops

gh secret set AZURE_WEBAPP_NAME --body "btc-mlops-api" --repo Nadhir1210/btc-mlops
```

---

## Étape 3: Vérifier les Secrets

```powershell
# Lister les secrets (ne montre pas les valeurs)
gh secret list --repo Nadhir1210/btc-mlops
```

Vous devriez voir:
```
AZURE_CREDENTIALS_V2
ACR_USERNAME
ACR_PASSWORD
ACR_LOGIN_SERVER
AZURE_WEBAPP_NAME
```

---

## Étape 4: Commit et Push les Workflows

```bash
git add .github/workflows/ci-cd-complete.yml
git add .github/workflows/drift-detection.yml
git commit -m "feat: Add new CI/CD pipelines v2 with updated secrets"
git push origin main
```

---

## Étape 5: Tester le Pipeline

1. Allez sur: **https://github.com/Nadhir1210/btc-mlops/actions**

2. Vous devriez voir le workflow **"Complete CI/CD Pipeline v2"** se lancer automatiquement

3. Ou déclenchez-le manuellement:
   - Cliquez sur **"Complete CI/CD Pipeline v2"**
   - Cliquez sur **"Run workflow"**
   - Sélectionnez **"main"**
   - Cliquez sur **"Run workflow"**

---

## 📊 Workflows Créés

### 1. Complete CI/CD Pipeline v2
- **Fichier**: `.github/workflows/ci-cd-complete.yml`
- **Déclencheur**: Push sur main/develop, PR, manuel
- **Jobs**: Tests → Training → Build → Deploy → Notification

### 2. Drift Detection & Monitoring
- **Fichier**: `.github/workflows/drift-detection.yml`
- **Déclencheur**: Cron quotidien (6h UTC), manuel
- **Jobs**: Drift check → Issue création → Retraining trigger

---

## 🔍 Vérification

### Vérifier les secrets
```powershell
gh secret list --repo Nadhir1210/btc-mlops
```

### Vérifier Azure
```powershell
# ACR
az acr show --name btcmlopsacr

# Web App
az webapp show --name btc-mlops-api --resource-group btc-mlops-prod

# Service Principal
az ad sp show --id 9ee30d29-deaa-49c8-8506-23b31c9ce671
```

### Test API
```powershell
# Health check
curl https://btc-mlops-api.azurewebsites.net/health

# Prediction
curl -X POST https://btc-mlops-api.azurewebsites.net/predict `
  -H "Content-Type: application/json" `
  -d '{"features": [50000, 51000, 49000, 50500, 1000, 50000000, 0.01, 0.02, 0.015, 0.01, 0.005, 0.02, 0.03, 0.025, 50000, 50200, 50300, 50400, 49900, 49800, 49700, 100, 200, 300, 1.02, 1.01, 60, 40, 50100, 50200, 0.5, 0.02, 0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.001, 0.002, 0.003, 1.01, 0.5]}'
```

---

## 🆘 Dépannage

### Erreur: "ACR login failed"
```powershell
# Vérifier et régénérer le password
az acr credential renew --name btcmlopsacr --password-name password
$newPassword = (az acr credential show --name btcmlopsacr --query "passwords[0].value" -o tsv)
gh secret set ACR_PASSWORD --body $newPassword --repo Nadhir1210/btc-mlops
```

### Erreur: "Azure login failed"
```powershell
# Vérifier le service principal
az ad sp show --id 9ee30d29-deaa-49c8-8506-23b31c9ce671

# Recréer si nécessaire
az ad sp create-for-rbac --name "btc-mlops-github-actions-v2" `
  --role contributor `
  --scopes /subscriptions/44be7a71-c968-4817-a5da-a30e18af4c15/resourceGroups/btc-mlops-prod `
  --sdk-auth
```

### Erreur: "Deployment failed"
```powershell
# Redémarrer le Web App
az webapp restart --name btc-mlops-api --resource-group btc-mlops-prod

# Vérifier les logs
az webapp log tail --name btc-mlops-api --resource-group btc-mlops-prod
```

---

## ✅ Checklist Finale

- [ ] ACR credentials récupérés
- [ ] 5 secrets créés sur GitHub
- [ ] Secrets vérifiés avec `gh secret list`
- [ ] Workflows commités et pushés
- [ ] Pipeline CI/CD testé avec succès
- [ ] Deployment Azure réussi
- [ ] Health check API OK
- [ ] Prediction endpoint testé

---

**Status**: Prêt pour production 🚀
