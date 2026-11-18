#!/usr/bin/env bash
# Publica o serviço de IA no Azure Container Instances (ACI), no mesmo Resource Group da API .NET.
#
#   1. Constrói a imagem com o Docker da sua máquina e envia ao ACR. Com --remoto, constrói
#      dentro da Azure (az acr build); assinaturas de estudante costumam bloquear isso.
#   2. Recria o container no ACI com a imagem nova e as variáveis do .env.cloud
#   3. Espera o /health responder
#
# Pré-requisitos: Docker Desktop aberto, az login e o .env.cloud preenchido (copie do .env.cloud.example).
# Uso (na raiz do repositório, no Git Bash): bash scripts/deploy_aci.sh [--remoto]
#
# Os segredos vão como "secure environment variables": o portal e o "az container show"
# não mostram o valor. Este script nunca imprime nenhum deles.

set -euo pipefail
export MSYS_NO_PATHCONV=1   # no Git Bash, impede que "/subscriptions/..." vire caminho do Windows

# Trabalha sempre na raiz do repositório, com caminhos relativos: o git e o docker são programas
# do Windows e não entendem "/c/Users/..." (o MSYS_NO_PATHCONV acima desliga a tradução).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_FILE=".env.cloud"

info() { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok: %s\n' "$*"; }
erro() { printf '\nERRO: %s\n' "$*" >&2; exit 1; }
az_tsv() { az "$@" -o tsv | tr -d '\r'; }

# ---------- Configuração ----------
[ -f "$ENV_FILE" ] || erro "crie o .env.cloud a partir do .env.cloud.example."
set -a
# shellcheck disable=SC1090
source <(tr -d '\r' < "$ENV_FILE")   # tr: tira o \r se o arquivo foi salvo no Bloco de Notas
set +a

for nome in AZ_RESOURCE_GROUP ACR_NAME ACI_NAME ACI_DNS_LABEL IA_API_KEY; do
    [ -n "${!nome:-}" ] || erro "preencha $nome no .env.cloud."
done
[ "${#IA_API_KEY}" -ge 16 ] || erro "IA_API_KEY precisa de 16 ou mais caracteres."
if [ -f .env ] && grep -qxF "IA_API_KEY=$IA_API_KEY" .env 2>/dev/null; then
    erro "a IA_API_KEY da nuvem é igual à do seu .env local. Gere outra (python -c \"import secrets; print(secrets.token_urlsafe(32))\")."
fi
ACI_CPU="${ACI_CPU:-1}"
ACI_MEMORY_GB="${ACI_MEMORY_GB:-2}"

# Variável só entra no container se tiver valor (as integrações vazias ficam desligadas).
PLAIN=(APP_ENV=cloud)
SECURE=("IA_API_KEY=$IA_API_KEY")
for nome in DOTNET_API_URL IA_USERNAME BLOB_CONTAINER BLOB_PREFIX \
            ISSUE_TYPE_ID_BURACO ISSUE_TYPE_ID_FAIXA_APAGADA ISSUE_TYPE_ID_POSTE_APAGADO ISSUE_TYPE_ID_VIA_ALAGADA \
            LLM_BASE_URL LLM_MODEL LLM_FALLBACK_BASE_URL LLM_FALLBACK_MODEL LLM_TIMEOUT_SECONDS \n            SCHEDULER_ENABLED SCHEDULER_INTERVAL_MINUTES; do
    if [ -n "${!nome:-}" ]; then PLAIN+=("$nome=${!nome}"); fi
done
for nome in IA_PASSWORD BLOB_CONNECTION_STRING LLM_API_KEY LLM_FALLBACK_API_KEY; do
    if [ -n "${!nome:-}" ]; then SECURE+=("$nome=${!nome}"); fi
done

# ---------- Azure ----------
info "Login na Azure"
az account show >/dev/null 2>&1 || erro "você não está logado. Rode: az login"
if [ -n "${AZ_SUBSCRIPTION_ID:-}" ]; then az account set --subscription "$AZ_SUBSCRIPTION_ID"; fi
ok "assinatura: $(az_tsv account show --query name)"

az group show --name "$AZ_RESOURCE_GROUP" -o none 2>/dev/null \
    || erro "o Resource Group $AZ_RESOURCE_GROUP não existe (ele é criado pelo 01-provisionar.sh do gs2-devops)."
az acr show --name "$ACR_NAME" -o none 2>/dev/null || erro "o ACR $ACR_NAME não existe."
AZ_LOCATION="${AZ_LOCATION:-$(az_tsv group show --name "$AZ_RESOURCE_GROUP" --query location)}"

# Tag = commit atual. Avisa se há mudança sem commit (a imagem não bateria com o GitHub).
TAG="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    echo "    aviso: há mudanças sem commit; a imagem leva o que está na pasta agora."
    TAG="$TAG-local"
fi
IMAGE_REPO="sampatech-ia"

ACR_SERVER="$(az_tsv acr show --name "$ACR_NAME" --query loginServer)"

# O .dockerignore deixa de fora .env, .venv, data/ e o modelo de 114 MB (baixado no build).
if [ "${1:-}" = "--remoto" ]; then
    info "Build da imagem dentro da Azure ($IMAGE_REPO:$TAG): leva de 3 a 6 minutos"
    az acr build --registry "$ACR_NAME" --image "$IMAGE_REPO:$TAG" --image "$IMAGE_REPO:latest" \
        --file Dockerfile . --only-show-errors -o none
else
    docker info >/dev/null 2>&1 || erro "o Docker não está rodando. Abra o Docker Desktop (ou use --remoto)."
    info "Build local da imagem ($IMAGE_REPO:$TAG): a primeira vez leva de 3 a 6 minutos"
    docker build -t "$ACR_SERVER/$IMAGE_REPO:$TAG" -t "$ACR_SERVER/$IMAGE_REPO:latest" .
    info "Enviando ao ACR $ACR_SERVER"
    az acr login --name "$ACR_NAME" --only-show-errors
    docker push "$ACR_SERVER/$IMAGE_REPO:$TAG"
    docker push "$ACR_SERVER/$IMAGE_REPO:latest"
fi
ok "imagem no ACR"
ACR_USER="$(az_tsv acr credential show --name "$ACR_NAME" --query username)"
ACR_PASS="$(az_tsv acr credential show --name "$ACR_NAME" --query 'passwords[0].value')"

# Recriar é o jeito previsível no ACI (não dá para trocar a imagem de um container parado
# nem mudar a porta). O DNS continua o mesmo; só o IP pode mudar.
if az container show --resource-group "$AZ_RESOURCE_GROUP" --name "$ACI_NAME" -o none 2>/dev/null; then
    info "Removendo o container anterior"
    az container delete --resource-group "$AZ_RESOURCE_GROUP" --name "$ACI_NAME" --yes -o none
    ok "removido"
fi

info "Criando o container $ACI_NAME (${ACI_CPU} vCPU, ${ACI_MEMORY_GB} GB)"
az container create \
    --resource-group "$AZ_RESOURCE_GROUP" --name "$ACI_NAME" --location "$AZ_LOCATION" \
    --image "$ACR_SERVER/$IMAGE_REPO:$TAG" --os-type Linux \
    --cpu "$ACI_CPU" --memory "$ACI_MEMORY_GB" \
    --ports 8000 --ip-address Public --dns-name-label "$ACI_DNS_LABEL" \
    --restart-policy Always \
    --registry-login-server "$ACR_SERVER" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
    --environment-variables "${PLAIN[@]}" \
    --secure-environment-variables "${SECURE[@]}" \
    --only-show-errors -o none
unset ACR_PASS
ok "criado"

FQDN="$(az_tsv container show --resource-group "$AZ_RESOURCE_GROUP" --name "$ACI_NAME" --query ipAddress.fqdn)"
URL="http://$FQDN:8000"

info "Esperando o /health (os modelos carregam na subida)"
for tentativa in $(seq 1 30); do
    if body="$(curl -fsS --max-time 10 "$URL/health" 2>/dev/null)"; then
        ok "respondeu na tentativa $tentativa"
        echo "    $body"
        echo
        echo "    URL:     $URL"
        echo "    Swagger: $URL/docs"
        echo "    Para a .NET (Variable Group): AI_SERVICE_BASE_URL=$URL"
        exit 0
    fi
    sleep 10
done
echo "    O /health não respondeu em 5 minutos. Veja o log:"
echo "    az container logs --resource-group $AZ_RESOURCE_GROUP --name $ACI_NAME"
exit 1
