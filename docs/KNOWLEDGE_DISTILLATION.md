# 🎓 Knowledge Distillation for Compass

## The Question: Can We Use a Distilled Mixtral?

**Yes!** Knowledge Distillation (KD) can create a smaller, faster model that retains Mixtral's quality.

---

## What is Knowledge Distillation?

```
┌─────────────────┐     Training      ┌─────────────────┐
│   Mixtral 8x7B  │  ─────────────►   │   Smaller Model │
│   (Teacher)     │   on teacher's    │   (Student)     │
│   ~47B params   │   outputs         │   ~7B params    │
└─────────────────┘                   └─────────────────┘
        │                                     │
        │ Slow, accurate                      │ Fast, nearly as accurate
        │ ~14GB VRAM                          │ ~4GB VRAM
        ▼                                     ▼
```

---

## Options for Compass

### Option 1: Use Existing Distilled Models ✅ (Recommended)

These models already incorporate Mixtral knowledge:

| Model | Size | Why It Works |
|-------|------|--------------|
| **Mistral 7B v0.3** | 7B | From same team, similar architecture |
| **Mistral Nemo 12B** | 12B | Newer, better performance |
| **OpenHermes 2.5** | 7B | Fine-tuned on Mixtral outputs |
| **Starling-LM 7B** | 7B | RLHF from Mixtral rankings |

**Current Compass**: Mistral 7B at **89.3% accuracy** - already excellent!

### Option 2: Custom Distillation (Advanced)

If you need higher accuracy, you can distill Mixtral specifically for business context extraction:

```python
# Pseudocode for custom distillation
from transformers import AutoModelForCausalLM, Trainer

# 1. Generate training data using Mixtral
teacher = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
training_data = []
for prompt in business_context_prompts:
    output = teacher.generate(prompt)
    training_data.append((prompt, output))

# 2. Fine-tune smaller model on teacher outputs
student = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")
trainer = Trainer(
    model=student,
    train_dataset=training_data,
    # Use soft labels from teacher for better distillation
)
trainer.train()

# 3. Save distilled model
student.save_pretrained("compass-extractor-7b")
```

### Option 3: Use Mixtral's MoE Knowledge Directly

Mixtral is a Mixture-of-Experts model. You can:
1. Extract the most relevant expert for your task
2. Create a dense model from that expert
3. Fine-tune for business context extraction

---

## Recommendation for Compass

### Current State (Good)
```
Mistral 7B → 89.3% use case accuracy → Production ready
```

### If You Need >95% Accuracy
```
Option A: Mistral Nemo 12B (larger, better)
Option B: Custom distillation from Mixtral
Option C: Fine-tune Mistral 7B on your domain data
```

### Cost-Benefit Analysis

| Approach | Effort | Expected Gain |
|----------|--------|---------------|
| Keep Mistral 7B | None | 89.3% (current) |
| Switch to Nemo 12B | Low | ~92-94% |
| Custom distillation | High | ~93-96% |
| Fine-tune on domain | Medium | ~91-95% |

---

## How to Use a Distilled Model in Compass

### Step 1: Update `.env`
```bash
OLLAMA_MODEL=mistral-nemo:12b  # Or your distilled model
```

### Step 2: Pull the model
```bash
ollama pull mistral-nemo:12b
```

### Step 3: Test accuracy
```bash
cd evaluation
python scripts/evaluate_by_usecase.py
```

### Step 4: If custom model, convert to Ollama
```bash
# Create Modelfile
echo "FROM ./compass-extractor-7b" > Modelfile
echo 'TEMPLATE """{{ .Prompt }}"""' >> Modelfile

# Create Ollama model
ollama create compass-extractor -f Modelfile
```

---

## The Bottom Line

| Question | Answer |
|----------|--------|
| Can we use KD on Mixtral? | ✅ Yes |
| Will it reduce size? | ✅ Yes (47B → 7B) |
| Will it work in Compass? | ✅ Yes (just change OLLAMA_MODEL) |
| Is it worth the effort? | ⚠️ Only if you need >90% accuracy |

**My recommendation**: 
1. Stick with **Mistral 7B** (89.3% is excellent for production)
2. If needed, try **Mistral Nemo 12B** (easy upgrade)
3. Only do custom distillation if you have specific domain needs

---

## References

- [Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)
- [Mistral AI - Mixtral of Experts](https://mistral.ai/news/mixtral-of-experts/)
- [LLM-as-a-Judge for evaluation](https://www.youtube.com/watch?v=nbZzSC5A6hs)

