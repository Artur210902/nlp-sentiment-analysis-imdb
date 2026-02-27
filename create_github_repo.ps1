# Script to create GitHub repository and push code
# This script requires GitHub CLI (gh) or manual repository creation

$repoName = "nlp-sentiment-analysis-imdb"
$username = "Artur210902"
$repoUrl = "https://github.com/$username/$repoName.git"

Write-Host "Checking if repository exists..."
$response = curl -s -o $null -w "%{http_code}" "https://github.com/$username/$repoName"

if ($response -eq "200") {
    Write-Host "Repository already exists!"
    Write-Host "Adding remote and pushing code..."
    git remote add origin $repoUrl 2>&1 | Out-Null
    git branch -M main
    git push -u origin main
} else {
    Write-Host "Repository does not exist. Please create it manually on GitHub:"
    Write-Host "1. Go to https://github.com/new"
    Write-Host "2. Repository name: $repoName"
    Write-Host "3. Description: NLP Final Project: Sentiment Analysis on IMDB Dataset using Hybrid Approach"
    Write-Host "4. Set to Public"
    Write-Host "5. DO NOT initialize with README, .gitignore, or license"
    Write-Host "6. Click 'Create repository'"
    Write-Host ""
    Write-Host "After creating, run this script again or execute:"
    Write-Host "  git remote add origin $repoUrl"
    Write-Host "  git branch -M main"
    Write-Host "  git push -u origin main"
}
