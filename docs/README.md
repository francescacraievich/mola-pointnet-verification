# PointNet Verification Report

> Formal Verification of PointNet Neural Networks using α,β-CROWN

---

## Abstract

Questo report presenta un'analisi di verifica formale per reti neurali PointNet applicate alla classificazione di point cloud. Utilizziamo α,β-CROWN per verificare la robustezza del modello rispetto a perturbazioni dell'input.

---

## 1. Introduzione

### 1.1 Contesto

Le reti neurali PointNet sono ampiamente utilizzate per l'elaborazione di point cloud 3D. La verifica formale di queste reti è essenziale per applicazioni safety-critical come la guida autonoma.

### 1.2 Obiettivi

- Verificare la robustezza locale del modello PointNet
- Analizzare le proprietà di sicurezza della rete
- Quantificare i bound di perturbazione ammissibili

---

## 2. Metodologia

### 2.1 Architettura PointNet

```python
# Architettura semplificata per verifica
class PointNetForVerification(nn.Module):
    def __init__(self, n_points=64, num_classes=2):
        # MLP: 3 -> 64 -> 128 -> 256
        # MaxPool globale
        # Classifier: 256 -> 128 -> 64 -> num_classes
```

### 2.2 α,β-CROWN

α,β-CROWN è un framework state-of-the-art per la verifica di reti neurali che utilizza:

- **Linear Relaxation**: rilassamento lineare dei vincoli non-lineari
- **Branch and Bound**: per raffinare i bound
- **GPU Acceleration**: per scalare a reti più grandi

### 2.3 Configurazione della Verifica

| Parametro | Valore |
|-----------|--------|
| Epsilon (ε) | 0.01 |
| Numero punti | 64 |
| Timeout | 300s |
| Metodo | α,β-CROWN |

---

## 3. Dataset

### 3.1 Preparazione dei Dati

I dati sono estratti da point cloud LiDAR e preprocessati per:

1. **Normalizzazione**: centramento e scaling
2. **Sampling**: selezione di 64 punti per gruppo
3. **Feature extraction**: coordinate (x, y, z)

### 3.2 Classi

- **Classe 0**: Regioni non critiche
- **Classe 1**: Regioni critiche (ostacoli, pedoni, ecc.)

---

## 4. Risultati

### 4.1 Accuratezza del Modello

| Metrica | Valore |
|---------|--------|
| Training Accuracy | XX.X% |
| Test Accuracy | XX.X% |
| Loss finale | X.XXX |

### 4.2 Risultati della Verifica

<!-- TODO: inserire risultati reali -->

```
Verification Results:
- Verified: XX samples
- Falsified: XX samples
- Unknown: XX samples
- Verification Rate: XX.X%
```

### 4.3 Analisi dei Bound

<!-- TODO: inserire grafici -->

---

## 5. Discussione

### 5.1 Interpretazione dei Risultati

I risultati mostrano che...

### 5.2 Limitazioni

- Scalabilità a point cloud più grandi
- Gestione del MaxPool nella verifica
- Trade-off tra precisione e tempo di verifica

---

## 6. Conclusioni

Questo lavoro ha dimostrato la fattibilità della verifica formale per reti PointNet. I risultati indicano che...

### 6.1 Lavori Futuri

- Estensione a PointNet++
- Verifica di proprietà globali
- Integrazione con pipeline di guida autonoma

---

## Riferimenti

1. Qi, C. R., et al. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." CVPR 2017.
2. Wang, S., et al. "Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification." NeurIPS 2021.
3. Zhang, H., et al. "General Cutting Planes for Bound-Propagation-Based Neural Network Verification." NeurIPS 2022.

---

<p align="center">
  <i>Report generato per il progetto mola-pointnet-verification</i>
</p>
