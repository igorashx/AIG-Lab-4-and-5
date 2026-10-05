## Laboratoarele 4–5: Modele de difuzie

## Sep 28, 2026

Două laboratoare de câte 2 ședințe: în Laboratorul 4 implementați un model DDPM de la zero, în Laboratorul 5 măsurați efectul setărilor unui model Stable Diffusion preantrenat.

## Informații generale

Laboratoarele 4 și 5 continuă seria (1: MLP și LSTM, 2: autoencoder și VAE, 3: GAN). Lucrați individual sau în echipe de câte 2, în Google Colab cu GPU (Runtime → Change runtime type → T4 GPU).

Folosim PyTorch, nu Keras ca la laboratorul GAN: biblioteca diffusers din Laboratorul 5 este construită pe PyTorch, iar Laboratorul 4 vă pregătește pentru ea.

|   | Laboratorul 4  | Laboratorul 5  |
| --- | --- | --- |
| Temă Durată | DDPM de la zero 2 ședințe (180 min) | Stable Diffusion în practică 2 ședințe (180 min) |
| Biblioteci | torch, torchvision, matplotlib | diffusers, transformers, accelerate, torchmetrics
Date / model | Fashion-MNIST | Stable Diffusion 1.5, preantrenat |
| Termen de predare | 12.10.2026  | 12.10.2026 |

## La fiecare laborator predați:

- 1. Notebook-ul .ipynb rulat complet, cu toate rezultatele vizibile.

- 2. Un raport PDF de 2–4 pagini: figurile cerute, tabelele, răspunsurile la întrebările de control și concluziile voastre.

Reguli pentru experimente: schimbați un singur parametru o dată și fixați seed-ul ( torch.manual_seed(42) ), altfel comparațiile nu au valoare. Salvați modelele după antrenare ( torch.save ), ca să nu pierdeți munca dacă sesiunea Colab se închide.

Notații folosite în formule: √ = radical; ᾱₜ = produsul α₁·α₂·…·αₜ; N(0, I) = zgomot gaussian standard; ε_θ(xₜ, t) = zgomotul prezis de rețea.


## Laboratorul 4 — DDPM de la zero

Obiectiv: implementați procesul direct, antrenarea și generarea unui model DDPM pe imagini Fashion-MNIST, comparați două programe de zgomot și măsurați compromisul dintre numărul de pași și calitate.

La final veți ști să: construiți un program de zgomot; explicați de ce rețeaua prezice zgomotul; antrenați și folosiți un U-Net mic; comparați eșantionarea DDPM cu DDIM; justificați alegerea programului de zgomot.

| Sarcina Conținut |   | Timp |
| --- | --- | --- |
| 1 | Datele | 10 min |
| 2 | Procesul direct | 30 min |
| 3 | Rețeaua (cod dat) | 15 min |
| 4 | Antrenarea cu programul liniar | 30 min |
| 5 | Generarea DDPM | 25 min |
| 6 | Mai puțini pași cu DDIM | 35 min |
| 7 | Liniar vs. cosinus | 25 min |
|   | Rezervă | 10 min |

În cod, pașii sunt indexați de la 0 la 999 (T = 1000).

## Sarcina 1. Datele (10 min)

- 1. Încărcați Fashion-MNIST cu torchvision.datasets.FashionMNIST .

- 2. Măriți imaginile la 32×32 ( transforms.Pad(2) ) și scalați pixelii în intervalul [−1, 1].

- 3. Creați un DataLoader cu loturi de 128 și afișați 16 imagini într-o grilă 4×4.

## Sarcina 2. Procesul direct (30 min)

Implementați două programe de zgomot, cu T = 1000:

- liniar: β crește uniform de la 0,0001 la 0,02 ( torch.linspace );

- cosinus (Nichol & Dhariwal, 2021), cu s = 0,008:


```
f(t) = cos²( ((t/T + s) / (1 + s)) · π/2 ), t = 0, 1, …, T
ᾱₜ = f(t) / f(0)
βₜ = 1 − ᾱₜ / ᾱₜ₋₁ (limitat superior la 0,999)
```

Pentru fiecare program calculați αₜ = 1 − βₜ și produsul cumulat ᾱₜ ( torch.cumprod ).

Scrieți funcția q_sample(x0, t, eps) , care produce direct imaginea zgomotoasă la pasul t:

```
xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε, unde ε ~ N(0, I)
```

Ce aveți de arătat:

- 1. O grilă cu aceeași imagine la t = 0, 50, 100, 250, 500, 750, 999, pentru ambele programe (două rânduri).

- 2. Un grafic cu √ᾱₜ în funcție de t, pentru ambele programe, pe același grafic.

- 3. Pasul t la care √ᾱₜ scade sub 0,5 pentru fiecare program.

Sarcina 3. Rețeaua (15 min, cod dat)

Copiați codul de mai jos într-o celulă. Este un U-Net mic, cu 3 niveluri (32, 64, 128 de canale), blocuri reziduale, conexiuni skip prin concatenare și încorporare sinusoidală a pasului t.

```
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def time_embedding(t, dim=128):
# t: tensor (B,) cu pașii; întoarce (B, dim)
half = dim // 2
/ half)
freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device)
args = t.float()[:, None] * freqs[None, :]
return torch.cat([torch.sin(args), torch.cos(args)], dim=1)
class ResBlock(nn.Module):
def __init__(self, c_in, c_out, t_dim=128):
super().__init__()
self.norm1 = nn.GroupNorm(8, c_in)
self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
self.t_proj = nn.Linear(t_dim, c_out) # pasul t intră aici
self.norm2 = nn.GroupNorm(8, c_out)
self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else
nn.Identity()
def forward(self, x, temb):
h = self.conv1(F.silu(self.norm1(x)))
h = h + self.t_proj(temb)[:, :, None, None]
h = self.conv2(F.silu(self.norm2(h)))
return h + self.skip(x) # conexiune reziduală
class UNet(nn.Module):
def __init__(self, ch=(32, 64, 128), t_dim=128):
super().__init__()
self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(),
nn.Linear(t_dim, t_dim))
self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
self.down1 = ResBlock(ch[0], ch[0])
self.down2 = ResBlock(ch[0], ch[1])
self.down3 = ResBlock(ch[1], ch[2])
self.mid = ResBlock(ch[2], ch[2])
self.up3 = ResBlock(2 * ch[2], ch[1])
self.up2 = ResBlock(2 * ch[1], ch[0])
self.up1 = ResBlock(2 * ch[0], ch[0])
self.out = nn.Conv2d(ch[0], 1, 1)
def forward(self, x, t):
temb = self.t_mlp(time_embedding(t))
x = self.inp(x)
s1 = self.down1(x, temb) # 32×32
s2 = self.down2(F.avg_pool2d(s1, 2), temb) # 16×16
s3 = self.down3(F.avg_pool2d(s2, 2), temb) # 8×8
h = self.mid(F.avg_pool2d(s3, 2), temb) # 4×4
up = lambda z: F.interpolate(z, scale_factor=2)
h = self.up3(torch.cat([up(h), s3], dim=1), temb) # skip:
concatenare
h = self.up2(torch.cat([up(h), s2], dim=1), temb)
h = self.up1(torch.cat([up(h), s1], dim=1), temb)
return self.out(h) # zgomotul
prezis
```

## Ce aveți de făcut:


- 1. Verificați că ieșirea are forma intrării: UNet()(torch.randn(8, 1, 32, 32), torch.randint(0, 1000, (8,))) → (8, 1, 32, 32).

- 2. Afișați numărul de parametri (așteptat: aproximativ 1 milion).

- 3. Marcați în notebook, prin comentarii, unde intră pasul t, unde sunt conexiunile skip și unde sunt conexiunile reziduale.

Sarcina 4. Antrenarea cu programul liniar (30 min)

La fiecare lot: alegeți t aleator între 0 și 999, generați ε, calculați xₜ cu q_sample , iar rețeaua prezice ε. Pierderea este eroarea pătratică medie:

```
L = media( ‖ ε − ε_θ(xₜ, t) ‖² ) (F.mse_loss)
```

Setări: Adam cu rata de învățare 0,0002, loturi de 128, 15 epoci (circa 5–10 minute pe T4). Salvați pierderea medie pe fiecare epocă, trasați graficul și salvați modelul ( torch.save ).

## Sarcina 5. Generarea DDPM (25 min)

Implementați eșantionarea DDPM (Ho et al., 2020, algoritmul 2), cu σₜ² = βₜ. Porniți de la zgomot pur și parcurgeți pașii t = 999, 998, …, 0. La fiecare pas:

```
xₜ₋₁ = (1 / √αₜ) · ( xₜ − (βₜ / √(1 − ᾱₜ)) · ε_θ(xₜ, t) ) + σₜ · z, z ~
N(0, I)
```

La ultimul pas (t = 0) nu adăugați zgomot (z = 0). Folosiți torch.no_grad() și

model.eval() .

Generați o grilă de 64 de imagini și afișați 8 stări intermediare pentru o singură imagine (de la zgomot la rezultat).

## Sarcina 6. Mai puțini pași cu DDIM (35 min)

Implementați eșantionarea deterministă DDIM pe un șir de pași rari (de exemplu, torch.linspace(999, 0, n).long() ). Pentru fiecare pereche de pași consecutivi t → t′:

```
x̂₀ = ( xₜ − √(1 − ᾱₜ) · ε_θ(xₜ, t) ) / √ᾱₜ
xₜ′ = √ᾱₜ′ · x̂₀ + √(1 − ᾱₜ′) · ε_θ(xₜ, t)
```

La ultimul pas, rezultatul este direct x̂₀. Generați câte 64 de imagini cu 10, 20, 50, 200 și 1000 de pași, pornind de la același zgomot. Măsurați timpul ( time.perf_counter , cu torch.cuda.synchronize() înainte de fiecare citire) și completați tabelul:


| Metodă Pași Timp pentru 64 de imagini (s) Calitate vizuală (1–5) Observații |
| --- |
| DDPM 1000 |
| DDIM 1000 |
| DDIM 200 |
| DDIM 50 |
| DDIM 20 |
| DDIM 10 |

## Sarcina 7. Liniar vs. cosinus (25 min)

Antrenați un al doilea model, identic, cu programul cosinus (aceleași setări: 15 epoci, seed 42). Apoi:

- 1. Generați câte 64 de imagini cu ambele modele, cu DDIM la 50 și la 20 de pași, din același zgomot de pornire.

- 2. Puneți grilele alături și notați calitatea vizuală (1–5).

- 3. Trasați curbele de pierdere ale ambelor modele pe același grafic.

Atenție: valorile pierderii nu sunt direct comparabile între cele două programe, pentru că media se face peste niveluri de zgomot diferite. Comparați modelele după imaginile generate.

Bonus (+1 punct): antrenați o variantă mai mică a rețelei, UNet(ch=(16, 32, 64)) , și comparați numărul de parametri, timpul de antrenare și calitatea.

## Întrebări de control — Laboratorul 4

- 1. De ce procesul direct nu are parametri învățați? Ce rol are în antrenare?

- 2. De ce putem sări direct la pasul t, fără să parcurgem pașii anteriori?

- 3. De ce rețeaua primește și numărul pasului t? Ce s-ar întâmpla fără el?

- 4. De ce generarea e mult mai lentă decât un pas de antrenare?

- 5. Ce diferență vedeți între programul liniar și cel cosinus la Sarcinile 2 și 7? Cum o explicați?

- 6. Ce câștigați și ce pierdeți când reduceți pașii DDIM de la 1000 la 10?

- 7. De ce pierderile celor două modele de la Sarcina 7 nu se pot compara direct?


## Ce predați — Laboratorul 4

- Grila procesului direct (2 programe), graficul √ᾱₜ și pașii la care scade sub 0,5

- Numărul de parametri ai rețelei și codul comentat

- Graficul pierderii pe epoci

- Grila de 64 de imagini DDPM și șirul de stări intermediare

- Tabelul DDPM / DDIM completat

- Comparația liniar vs. cosinus: grile și curbele de pierdere

- Răspunsurile la cele 7 întrebări

## Laboratorul 5 — Stable Diffusion în practică

Obiectiv: măsurați, prin experimente controlate, cum influențează setările unui model text-imagine calitatea, potrivirea cu textul și timpul de generare.

La final veți ști să: folosiți biblioteca diffusers ; explicați ce face fiecare setare (pași, scara de ghidare, seed, prompt negativ, metoda de eșantionare); măsurați scorul CLIP și variația lui; faceți o recomandare argumentată cu date.

| Sarcina Conținut |   | Timp |
| --- | --- | --- |
| 1 | Pregătirea și imaginile de bază 25 min |   |
| 2 | Instrumentele de măsurare | 25 min |
| 3 | Experimentele A–E | 100 min |
| 4 | Recomandarea | 20 min |
|   | Rezervă | 10 min |

## Sarcina 1. Pregătirea (25 min)

- 1. Instalați diffusers , transformers , accelerate și torchmetrics .

- 2. Încărcați modelul Stable Diffusion 1.5 de pe Hugging Face, identificatorul stable- diffusion-v1-5/stable-diffusion-v1-5 , în precizie float16, pe GPU: StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-

diffusion-v1-5", torch_dtype=torch.float16).to("cuda") . Descărcarea durează

câteva minute. Dacă modelul nu e disponibil, cereți profesorului o alternativă.


- 3. Folosiți aceste 3 prompturi (în engleză, limba în care a fost antrenat modelul):

- P1: „a watercolor painting of a red fox in a snowy forest”

- P2: „a photo of a modern library interior, natural light”

- P3: „a cartoon robot reading a book, flat illustration”

- 4. Configurația de bază: 30 de pași, scara de ghidare 7,5, rezoluție 512×512, seed-urile 42, 7 și 123 ( torch.Generator("cuda").manual_seed(...) ). Generați cele 9 imagini de bază (3 prompturi × 3 seed-uri).

## Sarcina 2. Instrumentele de măsurare (25 min)

- 1. Scorul CLIP (potrivirea imaginii cu textul): folosiți clip_score din Scorul mai mare înseamnă potrivire mai bună. torchmetrics.functional.multimodal , cu modelul openai/clip-vit-base-patch16 .

- 2. Timpul pe imagine: time.perf_counter() , cu torch.cuda.synchronize() înainte de fiecare citire. Ignorați prima generare (încălzire).

- 3. Scrieți o funcție ruleaza(config) care generează cele 9 imagini și întoarce: scorul CLIP mediu, abaterea standard a scorului CLIP, timpul mediu pe imagine și grila de imagini.

Scorul CLIP variază de la o imagine la alta. O diferență între două configurații mai mică decât abaterea standard nu este concludentă: spuneți acest lucru în raport, în loc să trageți concluzii din ea.

## Sarcina 3. Experimentele (100 min)

Porniți mereu de la configurația de bază și schimbați doar parametrul experimentului.


| Experiment | Parametrul modificat | Valori de testat |
| --- | --- | --- |
| A. Ghidare | guidance_scale | 1; 3; 7,5; 12; 20 |
| B. Pași | num_inference_steps | 5; 10; 20; 50 |
| C. Metoda de eșantionare | pipe.scheduler | DDIM, DPM-Solver (multistep), Euler — fiecare la 10 și 30 de pași|
| D. Seed | generator | 4 seed-uri diferite pentru P1; apoi seed-ul 42 de două ori |
| E. Prompt negativ | negative_prompt | fără / „blurry, low quality, text, watermark” |


La experimentul D verificați numeric că aceeași configurație cu același seed dă aceeași imagine (diferența medie între pixeli). La C, schimbați metoda cu

X.from_config(pipe.scheduler.config) , unde X este DDIMScheduler , DPMSolverMultistepScheduler sau EulerDiscreteScheduler .

Opțional (+1 punct): cu StableDiffusionImg2ImgPipeline , porniți de la una dintre

imaginile voastre și testați strength = 0,3; 0,6; 0,9. Explicați legătura cu procesul direct din Laboratorul 4.

## Tabelul de rezultate

Completați câte un rând pentru fiecare valoare testată (exemplul de mai jos arată formatul):

| Experiment | Valoare | Scor CLIP (medie ± abatere) | Timp / imagine (s) | Observații vizuale |
| --- | --- | --- | --- | --- |
| Bază | 30 pași, ghidare 7,5 |   |   |   |
| A | ghidare 1 |   |   |   |
| A | ghidare 20 |   |   |   |
| B | 5 pași |   |   |   |

Pentru experimentele A și B, faceți câte un grafic: scorul CLIP mediu în funcție de parametru, cu abaterea standard ca bare de eroare.


## Sarcina 4. Recomandarea (20 min)

Pe baza datelor, propuneți două configurații și argumentați-le în 5–10 rânduri:

- una pentru calitate maximă (timpul nu contează);

- una pentru viteză, de exemplu un serviciu care trebuie să răspundă în sub 2 secunde.

## Întrebări de control — Laboratorul 5

- 1. Ce se întâmplă cu imaginea la ghidare 1 și la ghidare 20? De ce?

- 2. De la câți pași nu mai vedeți îmbunătățiri clare? Coincide cu scorul CLIP?

- 3. De ce metodele DPM-Solver și Euler se descurcă mai bine la puțini pași decât DDPM-ul din Laboratorul 4?

- 4. De ce același seed dă aceeași imagine? La ce folosește asta în practică?

- 5. Ce nu măsoară scorul CLIP? Dați un exemplu din imaginile voastre.

- 6. Ce riscuri etice sau juridice vedeți în folosirea unor astfel de modele într-o companie?

## Ce predați — Laboratorul 5

- Grila de bază (9 imagini) și grilele comparative pentru experimentele A–E

- Tabelul de rezultate completat și cele două grafice (A și B), cu bare de eroare

Cele două configurații recomandate, cu argumentare

- Răspunsurile la cele 6 întrebări

Reguli: nu generați imagini cu persoane reale, logo-uri sau conținut ofensator; respectați licența modelului.

## Barem de evaluare

Fiecare laborator se notează cu maximum 10 puncte, plus 1 punct bonus.

| Criteriu | Laboratorul 4 | Laboratorul 5 | Ce urmărim  |
| --- | --- | --- | --- |
| Cod funcțional | 3 p | 3 p | Notebook-ul rulează de la cap la coadă fără erori |
| Experimente corecte | 3 p | 3 p | Un singur parametru schimbat o dată, seed fixat, toate valorile cerute|


| Criteriu | Laboratorul 4 | Laboratorul 5 | Ce urmărim |
| --- | --- | --- | --- |
| Interpretare | 3 p | 3 p | Răspunsuri corecte la întrebări, concluzii susținute de figuri și tabele |
| Prezentare | 1 p | 1 p | Figuri etichetate (axe, unități, titluri), raport clar|
| Bonus | +1 p (rețea mai mică) | +1 p | Opțional (img2img)|

## Resurse

- Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models — algoritmii 1 și 2. [URL 🔗](https://arxiv.org/abs/2006.11239)

- Nichol, Dhariwal (2021). Improved Denoising Diffusion Probabilistic Models — programul cosinus. [URL 🔗](https://arxiv.org/abs/2102.09672)

- Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models — eșantionarea DDIM. [URL 🔗](https://arxiv.org/abs/2010.02502)

- Documentația diffusers — pipeline-uri, metode de eșantionare, evaluare. [URL 🔗](https://huggingface.co/docs/diffusers)

- Capitolul 5 din curs și prezentările de la curs.
