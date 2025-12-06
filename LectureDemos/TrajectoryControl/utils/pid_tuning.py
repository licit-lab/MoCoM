import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid


# ============================================================================
# FONCTIONS DE COÛT (OBJECTIFS D'OPTIMISATION)
# ============================================================================

def compute_metrics(x_log, x_ref, v_log, a_log, e_log):
    """Calcule plusieurs métriques de performance"""
    # Erreur quadratique moyenne (MSE)
    mse = np.mean(e_log**2)
    
    # Erreur absolue moyenne (MAE)
    mae = np.mean(np.abs(e_log))
    
    # Erreur intégrale absolue (IAE)
    iae = np.sum(np.abs(e_log))
    
    # Erreur intégrale quadratique (ISE)
    ise = np.sum(e_log**2)
    
    # Erreur intégrale temporelle absolue (ITAE)
    itae = np.sum(np.arange(len(e_log)) * np.abs(e_log))
    
    # Dépassement maximal
    overshoot = np.max(x_log[1:] - x_ref) if np.max(x_log[1:]) > np.max(x_ref) else 0
    
    # Variations de commande (pénalise les changements brusques)
    control_variation = np.sum(np.abs(np.diff(a_log)))
    
    return {
        'mse': mse,
        'mae': mae,
        'iae': iae,
        'ise': ise,
        'itae': itae,
        'overshoot': overshoot,
        'control_variation': control_variation
    }

def cost_function(params, ref_function, simulate_pid, dt=0.1, Tsim=800, 
                  weight_error=1.0, weight_overshoot=0.1, weight_control=0.01):
    """
    Fonction de coût à minimiser.
    Combine plusieurs critères avec pondération.
    """
    Kp, Ki, Kd = params
    
    # Pénalité pour paramètres négatifs
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e10
    
    try:
        x_log, v_log, a_log, e_log, x_ref = simulate_pid(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, Tsim=Tsim, ref_function=ref_function)
        metrics = compute_metrics(x_log, x_ref, v_log, a_log, e_log)
        
        # Fonction de coût composite
        cost = (weight_error * metrics['ise'] + 
                weight_overshoot * metrics['overshoot']**2 + 
                weight_control * metrics['control_variation'])
        
        return cost
    except:
        return 1e10

# ============================================================================
# MÉTHODE 1 : GRID SEARCH (RECHERCHE EXHAUSTIVE)
# ============================================================================

def grid_search_tuning(Kp_range, Ki_range, Kd_range, ref_function, simulate_pid, dt=0.1, Tsim=800):
    """
    Recherche exhaustive sur une grille de paramètres.
    """
    print("=" * 60)
    print("MÉTHODE 1: GRID SEARCH")
    print("=" * 60)
    
    param_grid = {
        'Kp': Kp_range,
        'Ki': Ki_range,
        'Kd': Kd_range
    }
    
    grid = ParameterGrid(param_grid)
    best_cost = float('inf')
    best_params = None
    results = []
    
    print(f"Testing {len(grid)} combinations...")
    
    for i, params in enumerate(grid):
        Kp, Ki, Kd = params['Kp'], params['Ki'], params['Kd']
        cost = cost_function([Kp, Ki, Kd], ref_function, simulate_pid, dt, Tsim)
        
        results.append({
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'cost': cost
        })
        
        if cost < best_cost:
            best_cost = cost
            best_params = (Kp, Ki, Kd)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(grid)}")
    
    print(f"\nBest parameters: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
    print(f"Best cost: {best_cost:.4f}")
    
    return best_params, best_cost, results

# ============================================================================
# MÉTHODE 2 : OPTIMISATION PAR GRADIENT (NELDER-MEAD)
# ============================================================================

def gradient_based_tuning(ref_function, simulate_pid, initial_guess, dt=0.1, Tsim=800):
    """
    Optimisation par méthode Nelder-Mead (sans gradient).
    """
    print("\n" + "=" * 60)
    print("MÉTHODE 2: NELDER-MEAD OPTIMIZATION")
    print("=" * 60)
    
    print(f"Initial guess: Kp={initial_guess[0]}, Ki={initial_guess[1]}, Kd={initial_guess[2]}")
    
    result = minimize(
        cost_function,
        initial_guess,
        args=(ref_function, simulate_pid, dt, Tsim),
        method='Nelder-Mead',
        options={'maxiter': 200, 'disp': True}
    )
    
    best_params = result.x
    best_cost = result.fun
    
    print(f"\nOptimized parameters: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
    print(f"Final cost: {best_cost:.4f}")
    
    return best_params, best_cost

# ============================================================================
# MÉTHODE 3 : ALGORITHME ÉVOLUTIONNAIRE (DIFFERENTIAL EVOLUTION)
# ============================================================================

def evolutionary_tuning(ref_function, simulate_pid, bounds, dt=0.1, Tsim=800):
    """
    Optimisation par algorithme évolutionnaire.
    """
    print("\n" + "=" * 60)
    print("MÉTHODE 3: DIFFERENTIAL EVOLUTION")
    print("=" * 60)
    
    result = differential_evolution(
        cost_function,
        bounds,
        args=(ref_function, simulate_pid, dt, Tsim),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )
    
    best_params = result.x
    best_cost = result.fun
    
    print(f"\nOptimized parameters: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
    print(f"Final cost: {best_cost:.4f}")
    
    return best_params, best_cost
