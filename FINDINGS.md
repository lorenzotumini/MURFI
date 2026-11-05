# Unlearning e fault sui pesi: sintesi dei risultati

Aggiornamento: **5 novembre 2026**. Questa sintesi raccoglie gli esperimenti e le
valutazioni sviluppati nella chat: classificatore digits e ResNet-18 su CIFAR-10.
Il seed 0 di CIFAR-10 ha completato una valutazione test con checkpoint congelato;
i seed 1 e 2 sono in training. Non abbiamo testato LLM o fault fisici sulle GPU.

## Risultati principali

1. **Bit flip mirati possono ripristinare il riconoscimento di una classe
   soppressa nei nostri classificatori sperimentali.** Anche escludendo il
   classificatore finale, il risultato digits resta elevato e quello CIFAR-10
   raggiunge il 34,2% sul test nel solo seed finora completato.
2. **I fault casuali non hanno prodotto recuperi selettivi sostanziali nelle
   campagne eseguite.** Abbiamo osservato soprattutto predizioni invariate,
   altri cambiamenti e, con più fault, maggiore danno.
3. **Accuratezza bassa sulla classe rimossa non dimostra cancellazione della
   conoscenza.** La nostra procedura incoraggia esplicitamente risposte errate:
   va descritta come una semplice baseline di soppressione mediante relabeling.
4. **La rilevanza pratica dipende dall'accesso dell'attaccante.** Il caso più
   interessante è la modifica del modello di una vittima senza normali permessi
   di scrittura. Non abbiamo dimostrato un vantaggio rispetto ad alternative
   disponibili con gli stessi privilegi.

## 1. Che cosa abbiamo effettivamente testato

Abbiamo usato le cifre manoscritte 8×8 di scikit-learn, con una rete FP32 di
64 ingressi, 64 unità nascoste ReLU e 10 uscite: **4.810 parametri**. I target
erano le cifre 0, 1 e 2, ciascuna con tre seed: nove configurazioni.

Per ogni configurazione abbiamo preparato:

- un **originale**, addestrato su tutte le classi;
- un modello **soppresso**, derivato dall'originale e aggiornato con etichette
  casuali sempre errate per la cifra da rimuovere, mantenendo corrette le altre;
- un **controllo riaddestrato da zero**, senza esempi di training della cifra
  rimossa, ma con la stessa architettura e tutte e dieci le uscite.

La loss è cross-entropy rispetto alle etichette assegnate. La scelta del
checkpoint soppresso richiedeva accuratezza della classe rimossa ≤10% e
accuratezza sulle altre classi entro cinque punti percentuali dall'originale.
Non si tratta di una riproduzione di SalUn né di un metodo di cancellazione
certificata.

La ripartizione era circa 60% training, 10% ricerca dei fault, 10% selezione
dei checkpoint e 20% test finale. Il test contiene circa **36 immagini per
cifra**. I seed condividono il dataset: non sono dataset indipendenti.

Per «accuratezza della classe rimossa» intendiamo il **recall**: fra le immagini
di quella cifra, quante ricevono l'etichetta corretta. L'accuratezza sulle
classi mantenute è la media delle accuratezze delle nove classi. Il tasso di
falsi positivi misura quante immagini delle altre classi ricevono erroneamente
l'etichetta rimossa.

## 2. Fault mirati: recupero elevato, con interpretazione ancora aperta

La ricerca usa gradienti ed esempi etichettati della classe rimossa per scegliere
bit flip utili. Non consulta i pesi originali per annullare esattamente gli
aggiornamenti di unlearning. Cerca un **recupero del comportamento**, con un
budget massimo di 32 flip.

| Media su nove configurazioni | Tutte le matrici modificabili | Classificatore finale bloccato |
|---|---:|---:|
| Recall classe rimossa prima → dopo | 0,9% → **94,1%** | 0,9% → **89,1%** |
| Accuratezza classi mantenute prima → dopo | 96,7% → 96,2% | 96,7% → 95,4% |
| Recall del controllo riaddestrato dopo la propria ricerca | 32,3% | **0,3%** |
| Flip selezionati | 4–8 | 14–32 |
| Configurazioni che soddisfano tutti i limiti sul test | 7/9 | 6/9 |

Il criterio di recupero selettivo era fissato a: almeno **+20 punti percentuali**
di recall della classe rimossa, perdita di accuratezza mantenuta ≤2 punti e
aumento dei falsi positivi ≤1 punto. Le medie non implicano che tutti i singoli
esperimenti abbiano rispettato questi limiti.

Il controllo è stato sottoposto alla stessa procedura di ricerca, con il medesimo
budget massimo e selezione indipendente del numero di flip. **Anche la ricerca
sul controllo vede esempi etichettati della cifra assente dal suo training.**
Il suo risultato misura quindi quanto la ricerca stessa possa introdurre
capacità di riconoscimento.

Il divario con il classificatore finale bloccato è coerente con la persistenza
di informazione utile nel modello soppresso. Non è però una dimostrazione causale:
nel controllo l'uscita della cifra rimossa non ha ricevuto esempi positivi durante
il training. Bloccarla può rendere il recupero intrinsecamente più difficile.
Inoltre la ricerca è una forma di ottimizzazione supervisionata, anche senza
normali aggiornamenti di training. Sono ammessi bit di segno ed esponente:
pochi bit non significano necessariamente piccole variazioni numeriche.

Dettagli: [report dei fault mirati](reports/laptop/CPU_RESULTS.md) e
[protocollo CPU](LAPTOP_EXPERIMENT.md).

### Estensione CIFAR-10 / ResNet-18, seed 0

Il modello usa la stessa semplice baseline di relabeling, non un metodo MU
pubblicato riprodotto. Originale e controllo riaddestrato usano 100 epoche;
la soppressione ne usa 15. Le classi rimaste sono nove e il target è airplane.

| Valutazione sul test ufficiale | Recall airplane | Accuratezza classi mantenute | Falsi positivi airplane |
|---|---:|---:|---:|
| Originale | 95,40% | 94,28% | 0,57% |
| Soppresso, prima dei fault | 0,00% | 94,57% | 0,00% |
| Soppresso, cinque flip e head escluso | **34,20%** | **93,23%** | **0,86%** |
| Controllo riaddestrato, checkpoint selezionato | 0,00% | 94,53% | 0,00% |

Si tratta di **342 immagini airplane corrette su 1.000**, con perdita di
accuratezza mantenuta di 1,33 punti: tutti i criteri prefissati passano sul test.
Nessuno dei 20 trial random a cinque flip recupera airplane. Tutti e cinque
i flip scelti alterano il bit di esponente 26, moltiplicando il peso per 256:
la modifica è sparsa ma non numericamente piccola.

La prima ricerca greedy aveva recuperato solo lo 0,4% sul test. Le configurazioni
esplorative successive sono state confrontate sulla validazione: il miglior
recall è salito al 35,4% con head escluso. Le alternative peggiori e le campagne
interrotte sono conservate nel report. Il limite di 96 validazioni complete ha
fermato la ricerca vincente a cinque flip; il budget era di 128 flip.

I checkpoint selezionati e la configurazione sono stati congelati e verificati
prima della valutazione test finale, senza rieseguire la ricerca. Tuttavia il
medesimo test era già stato visto nell'esperimento iniziale: **conferma congelata
su un test già esaminato**, non un nuovo dataset intatto. La ricerca usa esempi
etichettati airplane; non è una prova causale di sola conoscenza residua. Le
repliche seed 1/2 hanno protocollo fisso e risultati ancora pendenti.

Dettagli e JSON archiviati: [report CIFAR-10](reports/resnet/RESULTS.md).

## 3. Fault casuali: nessun recupero sostanziale osservato

La nuova campagna include **tutti i 153.920 bit dei parametri FP32**, bias
compresi. Non filtra variazioni grandi, NaN/Inf o risultati sfavorevoli.
Ogni trial riparte dal checkpoint pulito e mantiene la corruzione per l'intero
batch di test. Le stesse posizioni logiche sono applicate ai tre modelli di
ogni configurazione.

Risultati aggregati sui nove modelli soppressi:

| Flip per trial | Trial | Predizioni invariate | Danno a utilità o falsi positivi | Output non finito | Recuperi selettivi sostanziali |
|---|---:|---:|---:|---:|---:|
| 1 | 90.000 | 88,588% | 1,547% | 0,072% | **0** |
| 4 | 9.000 | 61,822% | 5,656% | 0,156% | **0** |
| 16 | 9.000 | 16,222% | 21,900% | 1,078% | **0** |

Le colonne non esauriscono tutti gli esiti: restano altri cambiamenti delle
predizioni e piccoli incrementi di recall con utilità preservata. Nei trial
a singolo bit, 89 fault hanno aumentato il recall della classe rimossa: 85
violavano almeno un limite di utilità/falsi positivi; i quattro rimanenti
correggevano **una sola immagine aggiuntiva**, circa 2,8 punti percentuali.

Includendo originali e controlli, la campagna conta **324.000 valutazioni di
modelli**, relative a 108.000 trial di fault accoppiati fra i tre tipi di modello.
Nessun tipo ha mostrato recuperi selettivi sostanziali secondo il criterio.
I casi non finiti sono inclusi nei denominatori, non scartati.

Per ogni checkpoint, zero eventi in 10.000 trial a singolo bit danno un limite
superiore binomiale esatto unilaterale al 95% di circa **0,030% per fault
iniettato**, nel modello di campionamento adottato. È un limite per singolo
checkpoint/condizione; non una garanzia simultanea per tutte le configurazioni.

Questa probabilità è condizionata a un errore non corretto che raggiunge i
parametri. **Non è una probabilità per inferenza, ora di GPU o vita del dispositivo.**
Non abbiamo misurato tassi fisici, distribuzioni reali delle celle vulnerabili,
ECC, scrubbing o tempi di residenza. I trial a 4 e 16 flip sono stress test di
corruzioni accumulate, non modelli calibrati di eventi naturali o burst fisici.

Il confronto random iniziale del report sui fault mirati usava filtri e budget
diversi. Le due campagne non vanno aggregate come se avessero lo stesso modello
di guasto.

Dettagli: [report random](reports/random-faults/RESULTS.md) e
[protocollo random](RANDOM_FAULTS.md). Sono archiviati checkpoint NPZ, log dei
trial e verifiche: 663 modelli con fault sono stati ricostruiti indipendentemente
e hanno riprodotto le metriche registrate; quattro test dedicati sono passati.

## 4. Le metriche di «ignoranza» richiedono cautela

Un modello che non conosce una classe non è obbligato a indovinare uniformemente.
Il 10% è l'accuratezza attesa di un predittore che sceglie uniformemente fra dieci
etichette, non una proprietà automatica dell'unlearning. Il nostro modello usa
argmax e deve sempre scegliere una cifra: non ha un'uscita «sconosciuto».

Nel controllo senza esempi della classe rimossa, predire quell'etichetta è sempre
un errore durante il training. L'uscita può quindi essere sfavorita e il recall
risultare vicino a zero anche senza precedenti esempi positivi. Nel modello
soppresso, invece, le etichette intenzionalmente errate incoraggiano direttamente
la mancata previsione della classe.

Un controllo illustrativo su un solo originale ha anche mostrato punteggi
softmax elevati per input estranei: immagine completamente bianca → cifra 2
con score 99,8%; una A disegnata manualmente → cifra 4 con score 97,5%.
Fra 1.000 immagini di rumore uniforme, il 57,5% aveva score massimo >90%.
Questi score non sono una misura affidabile dell'appartenenza alle classi.
Si tratta di esempi illustrativi, non di una valutazione sistematica degli
input fuori distribuzione. [Valori registrati](reports/laptop/ood-check.json).

Per il prossimo esperimento occorre confrontare il comportamento con controlli
riaddestrati, distribuzioni delle predizioni e rappresentazioni interne, oltre
all'accuratezza della classe. Accuratezza zero, incertezza e cancellazione
dell'influenza dei dati sono obiettivi diversi.

## 5. Rilevanza hardware e confronto con altre tecniche

La letteratura consultata documenta attacchi hardware mirati a modelli, ma
**noi abbiamo eseguito esclusivamente XOR software sui parametri**. Non abbiamo
indotto fault fisici, dimostrato la vulnerabilità delle GPU disponibili o
modellato guasti stuck-at nei MAC.

[PrisonBreak](https://arxiv.org/html/2412.07192v3) dimostra un caso GPU di fault
mirati contro l'allineamento di LLM, con precise assunzioni di co-locazione e
hardware. Non è una dimostrazione del nostro obiettivo di inversione
dell'unlearning. La possibilità di scegliere un bit nel modello non assicura
che quel bit sia fisicamente alterabile con la direzione richiesta.

La nostra valutazione degli scenari è:

- **Accesso libero ai pesi:** confronto obbligatorio con ripristino di un
  originale disponibile, fine-tuning e modifica del classificatore. Un minor
  numero di bit cambiati non dimostra minor costo totale.
- **Sola API di inferenza:** i fault hardware non sono direttamente disponibili;
  sono pertinenti gli attacchi sugli input consentiti dal servizio.
- **Codice non privilegiato sullo stesso hardware, senza accesso in scrittura
  al modello della vittima:** possibile scenario in cui un fault offre una
  capacità altrimenti assente, se l'hardware e la configurazione lo consentono.
- **Accesso fisico a un dispositivo protetto:** scenario distinto, con costi e
  limiti specifici di fault injection.

Anche eventuali exploit che consentano scritture arbitrarie sono alternative
da considerare, non solo attacchi sui pesi. Il vantaggio pratico della nostra
tecnica rispetto ad alternative con lo stesso accesso **resta da dimostrare**.

## 6. Prossimi passi prioritari

1. Verificare provenienza, preprocessing e compatibilità del possibile trio
   CIFAR-10 originale / SalUn / riaddestrato individuato nel
   [documento sui checkpoint candidati](CHECKPOINT_CANDIDATES.md). Finora non
   abbiamo scaricato o validato quei pesi.
2. Ripetere su un metodo pubblicato, con una definizione esplicita di unlearning
   e controlli che distinguano recupero residuo, riapprendimento supervisionato
   e semplice riparazione del classificatore.
3. Mantenere le due domande separate: efficacia dei fault mirati e frequenza degli
   esiti selettivi sotto fault casuali definiti prima dell'esperimento.
4. Introdurre vincoli hardware verificabili su posizioni, direzioni e corruzioni
   collaterali. Valutare separatamente i fault nei parametri e nella computazione.
5. Confrontare tecniche disponibili con gli stessi privilegi e dati, misurando
   successo, utilità, costo di ricerca, affidabilità e persistenza.

**La conclusione oggi sostenibile è che pochi bit flip scelti mediante ricerca
possono invertire almeno in parte la soppressione di una classe nei nostri modelli,
mentre fault uniformemente casuali non hanno prodotto recuperi sostanziali
nelle campagne eseguite. Cancellazione effettiva della conoscenza, generalità
del fenomeno e vantaggio pratico di un attacco hardware restano questioni aperte.**
