// Draft Pick System JavaScript
let currentDraftState = null;
let searchTimeout = null;

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateUI();
});

function setupEventListeners() {
    // Hero search with debounce
    const heroSearch = document.getElementById('heroSearch');
    if (heroSearch) {
        heroSearch.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                searchHeroes(this.value);
            }, 300);
        });

        heroSearch.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                clearTimeout(searchTimeout);
                searchHeroes(this.value);
            }
        });
    }
}

function startDraft() {
    const rank = document.getElementById('rankSelect').value;
    const firstBanTeam = document.getElementById('firstBanSelect').value;
    
    showLoading(true);
    
    fetch('/draft/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            rank: rank,
            first_ban_team: firstBanTeam
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            currentDraftState = data.state;
            
            // Update team headers to show first ban indicator
            updateTeamHeaders(firstBanTeam);
            
            updateUI();
            showDraftPhase();
        } else {
            showError('Error starting draft: ' + data.error);
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network error: ' + error.message);
    });
}

function updateTeamHeaders(firstBanTeam) {
    const teamHeader = document.getElementById('teamHeader');
    const enemyHeader = document.getElementById('enemyHeader');
    
    if (teamHeader && enemyHeader) {
        if (firstBanTeam === 'team') {
            teamHeader.innerHTML = '<i class="fas fa-shield-alt me-1"></i>Tim Kawan <span class="badge bg-warning text-dark">First Ban</span>';
            enemyHeader.innerHTML = '<i class="fas fa-skull me-1"></i>Tim Musuh';
        } else {
            teamHeader.innerHTML = '<i class="fas fa-shield-alt me-1"></i>Tim Kawan';
            enemyHeader.innerHTML = '<i class="fas fa-skull me-1"></i>Tim Musuh <span class="badge bg-warning text-dark">First Ban</span>';
        }
    }
}

function processDraftAction(heroName, actionType) {
    showLoading(true);
    
    fetch('/draft/action', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            hero_name: heroName,
            action_type: actionType
        })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            currentDraftState = data.state;
            updateUI();
            updateRecommendations(data.recommendations || []);
            
            // Clear search after action
            document.getElementById('heroSearch').value = '';
            document.getElementById('heroResults').innerHTML = '';
            
            // Check if draft is complete
            if (currentDraftState.phase === 'complete') {
                showFinalAnalysis();
            }
        } else {
            showError('Error: ' + data.error);
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network error: ' + error.message);
    });
}

function searchHeroes(query) {
    const resultsContainer = document.getElementById('heroResults');
    
    if (!query || query.trim().length === 0) {
        // Show some default heroes when no query
        fetch('/draft/search?q=')
        .then(response => response.json())
        .then(data => {
            console.log('Default search response:', data);
            if (data.success) {
                displayHeroResults(data.heroes);
            } else {
                resultsContainer.innerHTML = '<p class="text-muted">No heroes available</p>';
            }
        })
        .catch(error => {
            console.error('Default search error:', error);
            resultsContainer.innerHTML = '<p class="text-danger">Search error occurred</p>';
        });
        return;
    }
    
    console.log('Searching for heroes:', query);
    
    fetch(`/draft/search?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
        console.log('Search response:', data);
        if (data.success) {
            displayHeroResults(data.heroes);
        } else {
            console.error('Search failed:', data.error);
            resultsContainer.innerHTML = '<p class="text-danger">Error: ' + (data.error || 'Search failed') + '</p>';
        }
    })
    .catch(error => {
        console.error('Search error:', error);
        resultsContainer.innerHTML = '<p class="text-danger">Network error occurred</p>';
    });
}

function displayHeroResults(heroes) {
    const container = document.getElementById('heroResults');
    console.log('Displaying heroes in container:', container, 'Heroes:', heroes);
    
    if (!container) {
        console.error('heroResults container not found!');
        return;
    }
    
    container.innerHTML = '';
    
    if (!heroes || heroes.length === 0) {
        container.innerHTML = '<p class="text-muted">No heroes found</p>';
        return;
    }
    
    console.log('Rendering', heroes.length, 'heroes');
    
    heroes.forEach(heroName => {
        const actionType = currentDraftState?.phase === 'ban' ? 'ban' : 'pick';
        const buttonClass = actionType === 'ban' ? 'btn-outline-secondary' : 'btn-outline-primary';
        const actionText = actionType === 'ban' ? 'Ban' : 'Pick';
        
        const heroCard = document.createElement('div');
        heroCard.className = 'col-md-4 mb-2';
        heroCard.innerHTML = `
            <button class="btn ${buttonClass} w-100" onclick="processDraftAction('${heroName}', '${actionType}')">
                <strong>${heroName}</strong><br>
                <small>${actionText}</small>
            </button>
        `;
        container.appendChild(heroCard);
    });
}

function updateRecommendations(recommendations) {
    const container = document.getElementById('counterRecommendations');
    
    if (!recommendations || recommendations.length === 0) {
        let message = 'Counter recommendations akan muncul saat fase pick tim kawan';
        
        if (currentDraftState?.phase === 'ban') {
            if (currentDraftState?.turn === 'team') {
                message = '🛡️ Giliran tim kawan untuk ban hero';
            } else {
                message = '⚔️ Menunggu tim musuh ban hero';
            }
        } else if (currentDraftState?.phase === 'pick') {
            if (currentDraftState?.turn === 'team') {
                message = 'No counter recommendations available';
            } else {
                message = '⚔️ Menunggu tim musuh pick hero';
            }
        }
        
        container.innerHTML = `
            <p class="text-muted text-center">
                <i class="fas fa-info-circle me-1"></i>
                ${message}
            </p>
        `;
        return;
    }
    
    container.innerHTML = '<h6 class="text-success mb-3">🛡️ Recommended Counters:</h6>';
    
    recommendations.slice(0, 8).forEach(rec => {
        const recCard = document.createElement('div');
        recCard.className = 'mb-2';
        recCard.innerHTML = `
            <button class="btn btn-outline-success btn-sm w-100" onclick="processDraftAction('${rec.hero}', 'pick')">
                <div class="d-flex justify-content-between align-items-center">
                    <span><strong>${rec.hero}</strong></span>
                    <span class="badge bg-success">${rec.role}</span>
                </div>
                <small class="text-muted">vs ${rec.target}</small>
            </button>
        `;
        container.appendChild(recCard);
    });
}

function updateUI() {
    if (!currentDraftState) return;
    
    // Update status bar
    updateStatusBar();
    
    // Update draft board
    updateDraftBoard();
    
    // Update progress
    updateProgress();
}

function updateStatusBar() {
    const phaseElement = document.getElementById('currentPhase');
    const turnElement = document.getElementById('currentTurn');
    
    if (phaseElement && turnElement) {
        phaseElement.textContent = currentDraftState.phase.toUpperCase();
        phaseElement.className = `badge ${currentDraftState.phase === 'ban' ? 'bg-secondary' : 'bg-info'}`;
        
        let turnText, iconText;
        if (currentDraftState.turn === 'team') {
            turnText = 'TIM KAWAN';
            iconText = '🛡️ ';
        } else {
            turnText = 'TIM MUSUH';
            iconText = '⚔️ ';
        }
        
        turnElement.textContent = iconText + turnText;
        turnElement.className = `badge ${currentDraftState.turn === 'team' ? 'bg-primary' : 'bg-danger'}`;
    }
}

function updateDraftBoard() {
    // Update banned heroes
    updateHeroList('teamBans', currentDraftState.banned_heroes.filter((_, i) => 
        currentDraftState.ban_sequence[i] === 'team'), 'secondary', 'ban');
    updateHeroList('enemyBans', currentDraftState.banned_heroes.filter((_, i) => 
        currentDraftState.ban_sequence[i] === 'enemy'), 'secondary', 'ban');
    
    // Update picked heroes
    updateHeroList('teamPicks', currentDraftState.team_picks, 'primary', 'pick');
    updateHeroList('enemyPicks', currentDraftState.enemy_picks, 'danger', 'pick');
}

function updateHeroList(containerId, heroes, badgeClass, type) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    if (heroes.length === 0) {
        container.innerHTML = `<span class="text-muted">No ${type}s yet</span>`;
        return;
    }
    
    heroes.forEach(hero => {
        const badge = document.createElement('span');
        badge.className = `badge bg-${badgeClass} me-1 mb-1`;
        badge.textContent = hero;
        container.appendChild(badge);
    });
}

function updateProgress() {
    const progressBar = document.getElementById('draftProgress');
    if (!progressBar) return;
    
    const totalSteps = currentDraftState.ban_sequence.length + currentDraftState.pick_sequence.length;
    const currentStep = currentDraftState.banned_heroes.length + 
                       currentDraftState.team_picks.length + 
                       currentDraftState.enemy_picks.length;
    
    const progress = Math.round((currentStep / totalSteps) * 100);
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
}

function showDraftPhase() {
    document.getElementById('setupPhase').style.display = 'none';
    document.getElementById('draftPhase').style.display = 'block';
    document.getElementById('heroSelection').style.display = 'block';
    document.getElementById('finalAnalysis').style.display = 'none';
}

function showFinalAnalysis() {
    document.getElementById('heroSelection').style.display = 'none';
    
    fetch('/draft/analyze')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayAnalysis(data.analysis);
            document.getElementById('finalAnalysis').style.display = 'block';
        } else {
            // Show error but still display final analysis section
            console.error('Analysis error:', data.error);
            displayErrorAnalysis(data.error);
            document.getElementById('finalAnalysis').style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Analysis fetch error:', error);
        displayErrorAnalysis('Failed to fetch analysis data');
        document.getElementById('finalAnalysis').style.display = 'block';
    });
}

function displayErrorAnalysis(errorMessage) {
    const originalContent = document.getElementById('originalAnalysisContent');
    if (!originalContent) return;
    
    originalContent.innerHTML = `
        <div class="alert alert-warning">
            <h6><i class="fas fa-exclamation-triangle me-2"></i>Analisis Tidak Tersedia</h6>
            <p class="mb-2">${errorMessage}</p>
            <p class="mb-0 small">Silakan mulai draft baru untuk melihat analisis persentase kemenangan yang lengkap.</p>
        </div>
        <div class="mt-3">
            <button class="btn btn-primary" onclick="startNewDraft()">
                <i class="fas fa-plus me-1"></i>Mulai Draft Baru
            </button>
        </div>
    `;
}

function displayAnalysis(analysis) {
    console.log('Analysis data:', analysis);
    
    // Display win rate prediction if available
    if (analysis.win_rate_prediction !== undefined) {
        displayWinRatePrediction(analysis);
    }
    
    // Display detailed analysis if available
    if (analysis.detailed_analysis && analysis.detailed_analysis.length > 0) {
        displayDetailedAnalysis(analysis.detailed_analysis);
    }
    
    // Display original analysis content
    displayOriginalAnalysis(analysis);
}

function displayWinRatePrediction(analysis) {
    const winRateSection = document.getElementById('winRatePrediction');
    const winRatePercentage = document.getElementById('winRatePercentage');
    const winRateFactors = document.getElementById('winRateFactors');
    
    if (!winRateSection || !winRatePercentage || !winRateFactors) return;
    
    // Set win rate percentage with color coding
    const winRate = analysis.win_rate_prediction;
    winRatePercentage.textContent = `${winRate}%`;
    
    // Color code based on win rate
    winRatePercentage.className = 'display-4 fw-bold';
    if (winRate >= 65) {
        winRatePercentage.classList.add('text-success');
    } else if (winRate >= 50) {
        winRatePercentage.classList.add('text-warning');
    } else {
        winRatePercentage.classList.add('text-danger');
    }
    
    // Display win rate factors
    if (analysis.win_rate_factors) {
        let factorsHtml = '';
        analysis.win_rate_factors.forEach(factor => {
            const isPositive = factor.includes('+');
            const isNegative = factor.includes('-');
            const badgeClass = isPositive ? 'success' : isNegative ? 'danger' : 'secondary';
            factorsHtml += `<span class="badge bg-${badgeClass} me-1 mb-1">${factor}</span>`;
        });
        winRateFactors.innerHTML = factorsHtml;
    }
    
    winRateSection.style.display = 'block';
}

function displayDetailedAnalysis(detailedAnalysis) {
    const detailedSection = document.getElementById('detailedAnalysisSection');
    const detailedList = document.getElementById('detailedAnalysisList');
    
    if (!detailedSection || !detailedList) return;
    
    let html = '';
    detailedAnalysis.forEach(reason => {
        let iconClass = 'fas fa-info-circle text-info';
        if (reason.startsWith('✓')) {
            iconClass = 'fas fa-check-circle text-success';
        } else if (reason.startsWith('✗')) {
            iconClass = 'fas fa-times-circle text-danger';
        } else if (reason.startsWith('~')) {
            iconClass = 'fas fa-exclamation-triangle text-warning';
        }
        
        html += `<div class="mb-1"><i class="${iconClass} me-1"></i>${reason}</div>`;
    });
    
    detailedList.innerHTML = html;
    detailedSection.style.display = 'block';
}

function displayOriginalAnalysis(analysis) {
    const originalContent = document.getElementById('originalAnalysisContent');
    if (!originalContent) return;
    
    if (!analysis.team_picks || analysis.team_picks.length === 0) {
        originalContent.innerHTML = '<p class="text-muted">Analysis not available</p>';
        return;
    }
    
    const roleDistribution = Object.entries(analysis.roles)
        .map(([role, count]) => `${role}: ${count}`)
        .join(', ');
    
    originalContent.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Komposisi Tim</h6>
                <div class="mb-3">
                    ${analysis.team_picks.map(hero => 
                        `<span class="badge bg-primary me-1">${hero}</span>`
                    ).join('')}
                </div>
                <p><strong>Distribusi Role:</strong> ${roleDistribution}</p>
                <p><strong>Skor Komposisi:</strong> 
                    <span class="badge ${analysis.composition_score >= 80 ? 'bg-success' : 
                                       analysis.composition_score >= 60 ? 'bg-warning' : 'bg-danger'}">
                        ${analysis.composition_score}/100
                    </span>
                </p>
            </div>
            <div class="col-md-6">
                <h6>Tim Musuh</h6>
                <div class="mb-3">
                    ${analysis.enemy_picks.map(hero => 
                        `<span class="badge bg-danger me-1">${hero}</span>`
                    ).join('')}
                </div>
                <h6>Analisis Dasar</h6>
                <p class="text-muted">${analysis.analysis}</p>
            </div>
        </div>
    `;
}

function resetDraft() {
    if (confirm('Are you sure you want to reset the current draft?')) {
        fetch('/draft/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                currentDraftState = null;
                document.getElementById('setupPhase').style.display = 'block';
                document.getElementById('draftPhase').style.display = 'none';
                document.getElementById('heroSelection').style.display = 'none';
                document.getElementById('finalAnalysis').style.display = 'none';
                
                // Clear search
                document.getElementById('heroSearch').value = '';
                document.getElementById('heroResults').innerHTML = '';
                document.getElementById('counterRecommendations').innerHTML = `
                    <p class="text-muted text-center">
                        <i class="fas fa-info-circle me-1"></i>
                        Counter recommendations akan muncul saat fase pick tim kawan
                    </p>
                `;
            }
        })
        .catch(error => {
            console.error('Reset error:', error);
        });
    }
}

function startNewDraft() {
    resetDraft();
}

function exportDraft() {
    if (!currentDraftState) return;
    
    fetch('/draft/analyze')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const exportData = {
                timestamp: new Date().toISOString(),
                rank: currentDraftState.rank,
                draft_result: data.analysis
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ml_draft_${new Date().getTime()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    })
    .catch(error => {
        console.error('Export error:', error);
    });
}

// Utility functions
function showLoading(show) {
    const spinner = document.getElementById('draftLoadingSpinner');
    if (show) {
        spinner.classList.remove('d-none');
    } else {
        spinner.classList.add('d-none');
    }
}

function showError(message) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert-danger');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        <strong>Error:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of main content
    const main = document.querySelector('main');
    main.insertBefore(alert, main.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

// Auto-refresh draft state
function refreshDraftState() {
    if (!currentDraftState) return;
    
    fetch('/draft/state')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentDraftState = data.state;
            updateUI();
            updateRecommendations(data.recommendations || []);
        }
    })
    .catch(error => {
        console.error('State refresh error:', error);
    });
}

// Refresh state every 10 seconds during active draft
setInterval(() => {
    if (currentDraftState && currentDraftState.phase !== 'complete') {
        refreshDraftState();
    }
}, 10000);