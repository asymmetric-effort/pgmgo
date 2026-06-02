//go:build ignore
// +build ignore

// Generator for dataset CSV files. Run with: go run examples/datasets/gen/main.go
package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
)

func main() {
	dir := filepath.Join("examples", "datasets", "data")
	os.MkdirAll(dir, 0755)

	rng := rand.New(rand.NewSource(42))

	genAsia(rng, dir)
	genAlarm(rng, dir)
	genSachs(rng, dir)
	genCancer(rng, dir)
	genStudent(rng, dir)
	genSprinkler(rng, dir)
	genSurvey(rng, dir)
	genTitanic(rng, dir)

	fmt.Println("All datasets generated.")
}

func writeCSV(path string, headers []string, rows [][]string) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write(headers)
	for _, r := range rows {
		w.Write(r)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		panic(err)
	}
	fmt.Printf("Wrote %s (%d rows)\n", path, len(rows))
}

func itoa(v int) string { return strconv.Itoa(v) }

func bern(rng *rand.Rand, p float64) int {
	if rng.Float64() < p {
		return 1
	}
	return 0
}

func categorical(rng *rand.Rand, probs []float64) int {
	r := rng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

// Asia (Lauritzen & Spiegelhalter) Bayesian network
// asia -> tub; smoke -> {lung, bronc}; {tub, lung} -> either; either -> {xray, dysp}; bronc -> dysp
func genAsia(rng *rand.Rand, dir string) {
	n := 1000
	headers := []string{"asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		asia := bern(rng, 0.01)
		tub := 0
		if asia == 1 {
			tub = bern(rng, 0.05)
		} else {
			tub = bern(rng, 0.01)
		}
		smoke := bern(rng, 0.50)
		lung := 0
		if smoke == 1 {
			lung = bern(rng, 0.10)
		} else {
			lung = bern(rng, 0.01)
		}
		bronc := 0
		if smoke == 1 {
			bronc = bern(rng, 0.60)
		} else {
			bronc = bern(rng, 0.30)
		}
		either := 0
		if tub == 1 || lung == 1 {
			either = 1
		}
		xray := 0
		if either == 1 {
			xray = bern(rng, 0.98)
		} else {
			xray = bern(rng, 0.05)
		}
		dysp := 0
		if either == 1 && bronc == 1 {
			dysp = bern(rng, 0.90)
		} else if either == 1 && bronc == 0 {
			dysp = bern(rng, 0.70)
		} else if either == 0 && bronc == 1 {
			dysp = bern(rng, 0.80)
		} else {
			dysp = bern(rng, 0.10)
		}
		rows[i] = []string{itoa(asia), itoa(tub), itoa(smoke), itoa(lung), itoa(bronc), itoa(either), itoa(xray), itoa(dysp)}
	}
	writeCSV(filepath.Join(dir, "asia.csv"), headers, rows)
}

// Alarm (Burglary) network
// Burglary -> Alarm; Earthquake -> Alarm; Alarm -> {JohnCalls, MaryCalls}
func genAlarm(rng *rand.Rand, dir string) {
	n := 1000
	headers := []string{"Burglary", "Earthquake", "Alarm", "JohnCalls", "MaryCalls"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		burg := bern(rng, 0.001)
		eq := bern(rng, 0.002)
		alarm := 0
		if burg == 1 && eq == 1 {
			alarm = bern(rng, 0.95)
		} else if burg == 1 {
			alarm = bern(rng, 0.94)
		} else if eq == 1 {
			alarm = bern(rng, 0.29)
		} else {
			alarm = bern(rng, 0.001)
		}
		john := 0
		if alarm == 1 {
			john = bern(rng, 0.90)
		} else {
			john = bern(rng, 0.05)
		}
		mary := 0
		if alarm == 1 {
			mary = bern(rng, 0.70)
		} else {
			mary = bern(rng, 0.01)
		}
		rows[i] = []string{itoa(burg), itoa(eq), itoa(alarm), itoa(john), itoa(mary)}
	}
	writeCSV(filepath.Join(dir, "alarm.csv"), headers, rows)
}

// Sachs protein signaling network (simplified)
// 11 proteins, discretized to 3 levels
func genSachs(rng *rand.Rand, dir string) {
	n := 500
	headers := []string{"Raf", "Mek", "Plcg", "PIP2", "PIP3", "Erk", "Akt", "PKA", "PKC", "P38", "Jnk"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		// Root nodes
		plcg := categorical(rng, []float64{0.30, 0.40, 0.30})
		pkc := categorical(rng, []float64{0.40, 0.35, 0.25})
		// PIP3 depends on Plcg
		pip3Probs := [][]float64{
			{0.60, 0.30, 0.10},
			{0.25, 0.50, 0.25},
			{0.10, 0.30, 0.60},
		}
		pip3 := categorical(rng, pip3Probs[plcg])
		// PIP2 depends on Plcg, PIP3
		pip2Base := (plcg + pip3) // 0..4
		pip2 := 0
		if pip2Base <= 1 {
			pip2 = categorical(rng, []float64{0.60, 0.30, 0.10})
		} else if pip2Base <= 3 {
			pip2 = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			pip2 = categorical(rng, []float64{0.10, 0.30, 0.60})
		}
		// PKA depends on PKC
		pkaProbs := [][]float64{
			{0.50, 0.35, 0.15},
			{0.25, 0.45, 0.30},
			{0.10, 0.30, 0.60},
		}
		pka := categorical(rng, pkaProbs[pkc])
		// Raf depends on PKC, PKA
		rafBase := (pkc + pka)
		raf := 0
		if rafBase <= 1 {
			raf = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if rafBase <= 3 {
			raf = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			raf = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		// Mek depends on Raf, PKC, PKA
		mekBase := (raf + pkc + pka)
		mek := 0
		if mekBase <= 2 {
			mek = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if mekBase <= 4 {
			mek = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			mek = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		// Erk depends on Mek, PKA
		erkBase := (mek + pka)
		erk := 0
		if erkBase <= 1 {
			erk = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if erkBase <= 3 {
			erk = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			erk = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		// Akt depends on PKA, Erk
		aktBase := (pka + erk)
		akt := 0
		if aktBase <= 1 {
			akt = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if aktBase <= 3 {
			akt = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			akt = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		// P38 depends on PKC, PKA
		p38Base := (pkc + pka)
		p38 := 0
		if p38Base <= 1 {
			p38 = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if p38Base <= 3 {
			p38 = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			p38 = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		// Jnk depends on PKC, PKA
		jnkBase := (pkc + pka)
		jnk := 0
		if jnkBase <= 1 {
			jnk = categorical(rng, []float64{0.55, 0.30, 0.15})
		} else if jnkBase <= 3 {
			jnk = categorical(rng, []float64{0.20, 0.50, 0.30})
		} else {
			jnk = categorical(rng, []float64{0.10, 0.25, 0.65})
		}
		rows[i] = []string{itoa(raf), itoa(mek), itoa(plcg), itoa(pip2), itoa(pip3), itoa(erk), itoa(akt), itoa(pka), itoa(pkc), itoa(p38), itoa(jnk)}
	}
	writeCSV(filepath.Join(dir, "sachs.csv"), headers, rows)
}

// Cancer network: Pollution -> Cancer; Smoker -> Cancer; Cancer -> {Xray, Dyspnoea}
func genCancer(rng *rand.Rand, dir string) {
	n := 1000
	headers := []string{"Pollution", "Smoker", "Cancer", "Xray", "Dyspnoea"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		poll := bern(rng, 0.10)
		smoker := bern(rng, 0.30)
		cancer := 0
		if poll == 1 && smoker == 1 {
			cancer = bern(rng, 0.05)
		} else if poll == 1 {
			cancer = bern(rng, 0.03)
		} else if smoker == 1 {
			cancer = bern(rng, 0.02)
		} else {
			cancer = bern(rng, 0.001)
		}
		xray := 0
		if cancer == 1 {
			xray = bern(rng, 0.90)
		} else {
			xray = bern(rng, 0.20)
		}
		dysp := 0
		if cancer == 1 {
			dysp = bern(rng, 0.65)
		} else {
			dysp = bern(rng, 0.30)
		}
		rows[i] = []string{itoa(poll), itoa(smoker), itoa(cancer), itoa(xray), itoa(dysp)}
	}
	writeCSV(filepath.Join(dir, "cancer.csv"), headers, rows)
}

// Student network: D -> G; I -> {G, S}; G -> L
// D,I binary; G ternary; L,S binary
func genStudent(rng *rand.Rand, dir string) {
	n := 1000
	headers := []string{"D", "I", "G", "L", "S"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		d := bern(rng, 0.40)    // difficulty
		intl := bern(rng, 0.30) // intelligence
		// Grade depends on D and I
		g := 0
		if d == 0 && intl == 0 {
			g = categorical(rng, []float64{0.30, 0.40, 0.30})
		} else if d == 0 && intl == 1 {
			g = categorical(rng, []float64{0.05, 0.25, 0.70})
		} else if d == 1 && intl == 0 {
			g = categorical(rng, []float64{0.50, 0.35, 0.15})
		} else {
			g = categorical(rng, []float64{0.20, 0.40, 0.40})
		}
		// Letter depends on Grade
		l := 0
		if g == 2 {
			l = bern(rng, 0.90)
		} else if g == 1 {
			l = bern(rng, 0.40)
		} else {
			l = bern(rng, 0.10)
		}
		// SAT depends on Intelligence
		s := 0
		if intl == 1 {
			s = bern(rng, 0.80)
		} else {
			s = bern(rng, 0.40)
		}
		rows[i] = []string{itoa(d), itoa(intl), itoa(g), itoa(l), itoa(s)}
	}
	writeCSV(filepath.Join(dir, "student.csv"), headers, rows)
}

// Sprinkler network: Cloudy -> {Sprinkler, Rain}; {Sprinkler, Rain} -> WetGrass
func genSprinkler(rng *rand.Rand, dir string) {
	n := 1000
	headers := []string{"Cloudy", "Sprinkler", "Rain", "WetGrass"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		cloudy := bern(rng, 0.50)
		sprinkler := 0
		if cloudy == 1 {
			sprinkler = bern(rng, 0.10)
		} else {
			sprinkler = bern(rng, 0.50)
		}
		rain := 0
		if cloudy == 1 {
			rain = bern(rng, 0.80)
		} else {
			rain = bern(rng, 0.20)
		}
		wet := 0
		if sprinkler == 1 && rain == 1 {
			wet = bern(rng, 0.99)
		} else if sprinkler == 1 {
			wet = bern(rng, 0.90)
		} else if rain == 1 {
			wet = bern(rng, 0.90)
		} else {
			wet = bern(rng, 0.01)
		}
		rows[i] = []string{itoa(cloudy), itoa(sprinkler), itoa(rain), itoa(wet)}
	}
	writeCSV(filepath.Join(dir, "sprinkler.csv"), headers, rows)
}

// Survey data: Age -> {Education, Occupation}; Education -> {Occupation, Residence}; {Occupation, Residence} -> Transportation
// Age: 0=young, 1=adult, 2=old; Education: 0=high, 1=uni; Occupation: 0=emp, 1=self; Residence: 0=small, 1=big; Transportation: 0=car, 1=train, 2=other
func genSurvey(rng *rand.Rand, dir string) {
	n := 500
	headers := []string{"Age", "Education", "Occupation", "Residence", "Transportation"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		age := categorical(rng, []float64{0.30, 0.50, 0.20})
		edu := 0
		if age == 0 {
			edu = bern(rng, 0.60)
		} else if age == 1 {
			edu = bern(rng, 0.40)
		} else {
			edu = bern(rng, 0.20)
		}
		occ := 0
		if edu == 1 {
			occ = bern(rng, 0.30)
		} else {
			occ = bern(rng, 0.50)
		}
		res := 0
		if edu == 1 {
			res = bern(rng, 0.60)
		} else {
			res = bern(rng, 0.40)
		}
		trans := 0
		if occ == 0 && res == 1 {
			trans = categorical(rng, []float64{0.40, 0.45, 0.15})
		} else if occ == 0 {
			trans = categorical(rng, []float64{0.60, 0.20, 0.20})
		} else if res == 1 {
			trans = categorical(rng, []float64{0.50, 0.30, 0.20})
		} else {
			trans = categorical(rng, []float64{0.70, 0.15, 0.15})
		}
		rows[i] = []string{itoa(age), itoa(edu), itoa(occ), itoa(res), itoa(trans)}
	}
	writeCSV(filepath.Join(dir, "survey.csv"), headers, rows)
}

// Titanic: Class -> Survived; Sex -> Survived; Age -> Survived
// Class: 0=1st, 1=2nd, 2=3rd, 3=crew; Sex: 0=male, 1=female; Age: 0=child, 1=adult; Survived: 0/1
func genTitanic(rng *rand.Rand, dir string) {
	n := 800
	headers := []string{"Class", "Sex", "Age", "Survived"}
	rows := make([][]string, n)
	for i := 0; i < n; i++ {
		class := categorical(rng, []float64{0.15, 0.15, 0.35, 0.35})
		sex := bern(rng, 0.35) // 35% female
		age := bern(rng, 0.95) // 95% adult
		// Survival probability based on class, sex, age
		p := 0.20 // base
		if class == 0 {
			p += 0.30
		} else if class == 1 {
			p += 0.10
		} else if class == 2 {
			p -= 0.05
		} else {
			p -= 0.02
		}
		if sex == 1 {
			p += 0.35
		}
		if age == 0 {
			p += 0.20
		}
		if p > 0.95 {
			p = 0.95
		}
		if p < 0.05 {
			p = 0.05
		}
		survived := bern(rng, p)
		rows[i] = []string{itoa(class), itoa(sex), itoa(age), itoa(survived)}
	}
	writeCSV(filepath.Join(dir, "titanic.csv"), headers, rows)
}
