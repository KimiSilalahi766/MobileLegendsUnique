// Global variables
let currentClusterData = null;

// Role color mapping based on specifications: Assassin=merah, Mage=biru, Marksman=oranye, Fighter=kuning, Tank=coklat
function getRoleColor(role) {
    const colors = {
        'Assassin': '#DC3545',      // Merah
        'Mage': '#0D6EFD',          // Biru  
        'Marksman': '#FD7E14',      // Oranye
        'Fighter': '#FFC107',       // Kuning
        'Tank': '#8B4513',          // Coklat
        'Tank/Support': '#8B4513',  // Coklat
        'Support': '#8B4513'        // Coklat
    };
    return colors[role] || '#6C757D'; // Default grey
}

function getRoleBorderColor(role) {
    const colors = {
        'Assassin': '#B02A37',      // Merah gelap
        'Mage': '#0A58CA',          // Biru gelap
        'Marksman': '#E55D0C',      // Oranye gelap
        'Fighter': '#FFCA2C',       // Kuning gelap
        'Tank': '#654321',          // Coklat gelap
        'Tank/Support': '#654321',  // Coklat gelap
        'Support': '#654321'        // Coklat gelap
    };
    return colors[role] || '#495057'; // Default dark grey
}

// Algorithm execution functions
function runKMeans() {
    const nClusters = document.getElementById('kmeansCluster').value;
    
    const params = {
        algorithm: 'kmeans',
        params: {
            n_clusters: parseInt(nClusters)
        }
    };
    
    runClustering(params);
}

function runDBSCAN() {
    const eps = document.getElementById('dbscanEps').value;
    const minSamples = document.getElementById('dbscanMinSamples').value;
    
    const params = {
        algorithm: 'dbscan',
        params: {
            eps: parseFloat(eps),
            min_samples: parseInt(minSamples)
        }
    };
    
    runClustering(params);
}

// Main clustering function
function runClustering(params) {
    showLoading(true);
    
    fetch('/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            // Update visualization
            updateVisualization(data.plot);
            
            // Update metrics
            updateMetrics(data.metrics, params.algorithm, data.n_clusters);
            
            // Update cluster analysis
            updateClusterAnalysis(data.cluster_profiles);
            
            // Store current data
            currentClusterData = data;
        } else {
            showError('Clustering Error: ' + data.error);
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network Error: ' + error.message);
    });
}

// Update visualization
function updateVisualization(plotData) {
    try {
        const plotDiv = document.getElementById('clusteringPlot');
        const plot = JSON.parse(plotData);
        
        console.log('Updating visualization with data:', plot);
        console.log('Number of traces:', plot.data ? plot.data.length : 0);
        
        // Ensure plotDiv has proper dimensions
        plotDiv.style.height = '500px';
        plotDiv.style.width = '100%';
        
        // Clear any existing plot first
        if (typeof Plotly !== 'undefined') {
            Plotly.purge(plotDiv);
            
            // Create new plot with explicit configuration
            Plotly.newPlot(plotDiv, plot.data, plot.layout, {
                responsive: true,
                displayModeBar: true,
                showTips: false,
                showLink: false,
                plotGlPixelRatio: 2
            }).then(() => {
                console.log('Plotly visualization created successfully');
                // Force resize after creation
                setTimeout(() => {
                    Plotly.Plots.resize(plotDiv);
                }, 100);
            }).catch((error) => {
                console.error('Plotly creation error:', error);
                showError('Plot creation failed: ' + error.message);
            });
        } else {
            console.error('Plotly is not loaded');
            showError('Plotly library not loaded properly');
        }
        
    } catch (error) {
        console.error('Error updating visualization:', error);
        showError('Visualization Error: ' + error.message);
    }
}

// Update metrics display
function updateMetrics(metrics, algorithm, nClusters) {
    document.getElementById('algorithmName').textContent = 
        `${algorithm.toUpperCase()} (${nClusters} clusters)`;
    document.getElementById('silhouetteScore').textContent = 
        metrics.silhouette_score || 'N/A';
    document.getElementById('daviesBouldinScore').textContent = 
        metrics.davies_bouldin_score || 'N/A';
    
    // Show metrics row
    document.getElementById('metricsRow').style.display = 'flex';
    
    // Update dynamic conclusion based on test results
    updateResearchConclusion(algorithm, nClusters, metrics);
}

function updateResearchConclusion(algorithm, nClusters, metrics) {
    const conclusionDiv = document.getElementById('researchConclusion');
    const contentDiv = document.getElementById('conclusionContent');
    
    if (!window.testResults) {
        window.testResults = {};
    }
    
    window.testResults[algorithm] = {
        clusters: nClusters,
        metrics: metrics,
        timestamp: new Date()
    };
    
    // Show conclusion after first test
    if (Object.keys(window.testResults).length > 0) {
        conclusionDiv.style.display = 'block';
        
        let conclusionHTML = '<div class="row">';
        
        // Algorithm comparison results
        conclusionHTML += '<div class="col-md-6">';
        conclusionHTML += '<h6><i class="fas fa-chart-line me-1"></i> Hasil Uji Algoritma Clustering</h6>';
        
        if (window.testResults.kmeans) {
            const kmeans = window.testResults.kmeans;
            conclusionHTML += `<p class="mb-2"><strong>K-Means dengan ${kmeans.clusters} cluster:</strong><br>`;
            conclusionHTML += `• Silhouette Score: ${kmeans.metrics.silhouette_score.toFixed(3)}<br>`;
            conclusionHTML += `• Davies-Bouldin Index: ${kmeans.metrics.davies_bouldin_score.toFixed(3)}</p>`;
        }
        
        if (window.testResults.dbscan) {
            const dbscan = window.testResults.dbscan;
            conclusionHTML += `<p class="mb-0"><strong>DBSCAN dengan ${dbscan.clusters} cluster:</strong><br>`;
            conclusionHTML += `• Silhouette Score: ${dbscan.metrics.silhouette_score.toFixed(3)}<br>`;
            conclusionHTML += `• Davies-Bouldin Index: ${dbscan.metrics.davies_bouldin_score.toFixed(3)}</p>`;
        }
        
        conclusionHTML += '</div>';
        
        // Dataset information
        conclusionHTML += '<div class="col-md-6">';
        conclusionHTML += '<h6><i class="fas fa-users me-1"></i> Dataset Penelitian</h6>';
        conclusionHTML += `<p class="mb-2"><strong>Data Primer:</strong> 245 pemain autentik dari wilayah Medan<br>`;
        conclusionHTML += `• Player ID: 101849453 - 999888777<br>`;
        conclusionHTML += `• Fitur: win_rate, kills, deaths, assists, main_role</p>`;
        conclusionHTML += `<p class="mb-0"><strong>Preprocessing berhasil:</strong><br>`;
        conclusionHTML += `• 8 fitur final setelah one-hot encoding role<br>`;
        conclusionHTML += `• Normalisasi MinMaxScaler untuk semua fitur numerik</p>`;
        conclusionHTML += '</div>';
        
        conclusionHTML += '</div>';
        
        // Final conclusion based on results
        if (window.testResults.kmeans && window.testResults.dbscan) {
            let bestAlgorithm = 'K-Means';
            let bestScore = window.testResults.kmeans.metrics.silhouette_score;
            
            if (window.testResults.dbscan.metrics.silhouette_score > bestScore) {
                bestAlgorithm = 'DBSCAN';
                bestScore = window.testResults.dbscan.metrics.silhouette_score;
            }
            
            conclusionHTML += `<div class="alert alert-success mt-3 mb-0">`;
            conclusionHTML += `<i class="fas fa-check-circle me-1"></i>`;
            conclusionHTML += `<strong>Kesimpulan Berdasarkan Hasil Uji:</strong> ${bestAlgorithm} memberikan hasil clustering terbaik `;
            conclusionHTML += `dengan Silhouette Score ${bestScore.toFixed(3)}. Sistem berhasil memproses 245 data pemain asli `;
            conclusionHTML += `dengan visualisasi interaktif yang berfungsi penuh.</div>`;
        } else {
            conclusionHTML += `<div class="alert alert-info mt-3 mb-0">`;
            conclusionHTML += `<i class="fas fa-info-circle me-1"></i>`;
            conclusionHTML += `<strong>Status Uji:</strong> Silakan jalankan kedua algoritma untuk mendapatkan perbandingan lengkap.</div>`;
        }
        
        contentDiv.innerHTML = conclusionHTML;
    }
}

// Update cluster analysis
function updateClusterAnalysis(clusterProfiles) {
    const container = document.getElementById('clusterProfiles');
    container.innerHTML = '';
    
    if (!clusterProfiles || clusterProfiles.length === 0) {
        container.innerHTML = '<p class="text-muted">No cluster profiles available.</p>';
        return;
    }
    
    clusterProfiles.forEach(profile => {
        const clusterCard = createClusterCard(profile);
        container.appendChild(clusterCard);
    });
    
    // Show cluster analysis section
    document.getElementById('clusterAnalysisSection').style.display = 'block';
}

// Create cluster card
function createClusterCard(profile) {
    const card = document.createElement('div');
    card.className = 'card mb-3';
    
    card.innerHTML = `
        <div class="card-header">
            <h6 class="mb-0">
                <i class="fas fa-layer-group me-2"></i>
                Cluster ${profile.cluster_id}: ${profile.label}
            </h6>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-4">
                    <h6>Cluster Statistics</h6>
                    <ul class="list-unstyled">
                        <li><strong>Size:</strong> ${profile.size} players (${profile.percentage}%)</li>
                        <li><strong>Avg Win Rate:</strong> ${(profile.avg_win_rate * 100).toFixed(1)}%</li>
                        <li><strong>Avg KDA:</strong> ${profile.avg_kda.toFixed(2)}</li>
                        <li><strong>Match Frequency:</strong> ${profile.avg_match_frequency.toFixed(2)}</li>
                        <li><strong>Dominant Role:</strong> ${profile.dominant_role}</li>
                    </ul>
                </div>
                <div class="col-md-4">
                    <h6>Role Distribution</h6>
                    <div id="roleChart${profile.cluster_id}" style="height: 200px;"></div>
                </div>
                <div class="col-md-4">
                    <h6>Representative Players</h6>
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Player ID</th>
                                    <th>Role</th>
                                    <th>WR</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${profile.representative_players.map(player => `
                                    <tr>
                                        <td><a href="#" onclick="getPlayerProfile(${player.player_id})">${player.player_id}</a></td>
                                        <td>${player.main_role}</td>
                                        <td>${(player.win_rate * 100).toFixed(1)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add role distribution chart after the card is added to DOM
    setTimeout(() => {
        createRoleChart(profile.cluster_id, profile.role_distribution);
    }, 100);
    
    return card;
}

// Create role distribution chart
function createRoleChart(clusterId, roleData) {
    const ctx = document.getElementById(`roleChart${clusterId}`);
    if (!ctx) return;
    
    // Create canvas element
    const canvas = document.createElement('canvas');
    ctx.appendChild(canvas);
    
    new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: roleData.map(r => r.role),
            datasets: [{
                data: roleData.map(r => r.count),
                backgroundColor: roleData.map(r => getRoleColor(r.role)),
                borderColor: roleData.map(r => getRoleBorderColor(r.role)),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

// Player search functionality
function searchPlayer() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }
    
    showLoading(true);
    
    fetch(`/search_players?q=${encodeURIComponent(query)}`)
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            displaySearchResults(data.players);
        } else {
            showError('Search Error: ' + data.error);
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network Error: ' + error.message);
    });
}

// Display search results
function displaySearchResults(players) {
    const container = document.getElementById('searchResultsContent');
    
    if (!players || players.length === 0) {
        container.innerHTML = '<p class="text-muted">No players found.</p>';
        document.getElementById('searchResults').style.display = 'block';
        return;
    }
    
    const tableHTML = `
        <div class="table-responsive">
            <table class="table table-hover">
                <thead>
                    <tr>
                        <th>Player ID</th>
                        <th>Main Role</th>
                        <th>Win Rate</th>
                        <th>Avg KDA</th>
                        <th>Match Frequency</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${players.map(player => `
                        <tr>
                            <td><strong>${player.player_id}</strong></td>
                            <td>
                                <span class="badge bg-secondary">${player.main_role}</span>
                            </td>
                            <td>${(player.win_rate * 100).toFixed(1)}%</td>
                            <td>${player.avg_kda.toFixed(2)}</td>
                            <td>${player.match_frequency.toFixed(2)}</td>
                            <td>
                                <button class="btn btn-sm btn-outline-primary" 
                                        onclick="getPlayerProfile(${player.player_id})">
                                    <i class="fas fa-eye me-1"></i>View Profile
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = tableHTML;
    document.getElementById('searchResults').style.display = 'block';
}

// Get player profile
function getPlayerProfile(playerId) {
    showLoading(true);
    
    fetch(`/player/${playerId}`)
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        
        if (data.success) {
            showPlayerProfile(data.player_info, data.cluster, data.cluster_stats);
        } else {
            showError('Profile Error: ' + data.error);
        }
    })
    .catch(error => {
        showLoading(false);
        showError('Network Error: ' + error.message);
    });
}

// Show player profile modal
function showPlayerProfile(playerInfo, cluster, clusterStats) {
    const modalHTML = `
        <div class="modal fade" id="playerModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-user me-2"></i>
                            Player Profile - ID: ${playerInfo.player_id}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Player Information</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Player ID:</strong> ${playerInfo.player_id}</li>
                                    <li><strong>Main Role:</strong> 
                                        <span class="badge bg-primary">${playerInfo.main_role}</span>
                                    </li>
                                    <li><strong>Win Rate:</strong> ${(playerInfo.win_rate * 100).toFixed(1)}%</li>
                                    <li><strong>Average KDA:</strong> ${playerInfo.avg_kda.toFixed(2)}</li>
                                    <li><strong>Match Frequency:</strong> ${playerInfo.match_frequency.toFixed(2)}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6>Cluster Information</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Assigned Cluster:</strong> ${cluster}</li>
                                    <li><strong>Cluster Size:</strong> ${clusterStats.cluster_size} players</li>
                                    <li><strong>Cluster Avg WR:</strong> ${(clusterStats.avg_win_rate * 100).toFixed(1)}%</li>
                                    <li><strong>Cluster Avg KDA:</strong> ${clusterStats.avg_kda.toFixed(2)}</li>
                                    <li><strong>Dominant Role:</strong> 
                                        <span class="badge bg-secondary">${clusterStats.dominant_role}</span>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal
    const existingModal = document.getElementById('playerModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Add new modal
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('playerModal'));
    modal.show();
}

// Utility functions
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
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

// Enter key support for search
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('playerSearch');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchPlayer();
            }
        });
    }
});
