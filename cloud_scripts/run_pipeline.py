#!/usr/bin/env python3
"""
Script maestro para ejecutar el pipeline completo en OCI.
Úsalo desde OCI Run Command o SSH.

Modo de uso:
  python3 run_pipeline.py              # Pipeline completo (datos + entrenamiento)
  python3 run_pipeline.py --data-only  # Solo pipeline de datos
  python3 run_pipeline.py --train-only # Solo entrenamiento (requiere datos procesados)
"""

import os, sys, subprocess, time, json

BASE_DIR = '/opt/football-analysis'
VENV_PYTHON = os.path.join(BASE_DIR, '.venv', 'bin', 'python3')

if not os.path.exists(VENV_PYTHON):
    print(f"❌ Virtualenv no encontrado en {VENV_PYTHON}")
    print("Ejecuta: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
    sys.exit(1)

def run_script(name, script_path, *args):
    """Ejecuta un script de Python con el venv y mide tiempo."""
    print(f"\n{'=' * 60}")
    print(f"▶ Ejecutando: {name}")
    print(f"{'=' * 60}")
    
    cmd = [VENV_PYTHON, script_path] + list(args)
    start = time.time()
    
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    
    elapsed = time.time() - start
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ Stderr:\n{result.stderr}")
    
    print(f"⌛ Tiempo: {elapsed:.1f}s | Exit: {result.returncode}")
    return result.returncode == 0

if __name__ == '__main__':
    data_only = '--data-only' in sys.argv
    train_only = '--train-only' in sys.argv
    
    SCRIPTS_DIR = os.path.join(BASE_DIR, 'cloud_scripts')
    
    if not os.path.exists(SCRIPTS_DIR):
        os.makedirs(SCRIPTS_DIR, exist_ok=True)
    
    pipeline_script = os.path.join(SCRIPTS_DIR, 'data_pipeline.py')
    train_script = os.path.join(SCRIPTS_DIR, 'train_models.py')
    
    success = True
    
    if not train_only:
        # Check if scripts exist, if not try local
        if not os.path.exists(pipeline_script):
            # Maybe run directly from this directory
            pipeline_script = os.path.join(os.path.dirname(__file__), 'data_pipeline.py')
        
        if os.path.exists(pipeline_script):
            success &= run_script('Data Pipeline', pipeline_script)
        else:
            print(f"⚠️ Script no encontrado: {pipeline_script}")
            success = False
    
    if not data_only and success:
        qatar_file = os.path.join(BASE_DIR, 'data', 'processed', 'qatar_2022.csv')
        features_file = os.path.join(BASE_DIR, 'data', 'processed', 'matches_with_features.csv')
        
        if not os.path.exists(features_file) and not train_only:
            print("⚠️ Datos procesados no encontrados. Ejecuta primero el pipeline de datos.")
        else:
            if not os.path.exists(train_script):
                train_script = os.path.join(os.path.dirname(__file__), 'train_models.py')
            
            if os.path.exists(train_script):
                success &= run_script('Model Training', train_script)
            else:
                print(f"⚠️ Script no encontrado: {train_script}")
                success = False
    
    if success:
        # Show results
        results_file = os.path.join(BASE_DIR, 'models', 'results_summary.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            print(f"\n{'=' * 60}")
            print("🏆 RESULTADOS FINALES")
            print(f"{'=' * 60}")
            for name, info in sorted(results['models'].items(), 
                                     key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"  {name:30s} {info['accuracy']:.1%}")
    
    print(f"\n{'✅' if success else '❌'} Pipeline {'completado' if success else 'falló'}")
