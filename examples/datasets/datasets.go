// Package datasets provides well-known Bayesian network datasets as embedded
// CSV data, comparable to pgmpy's built-in datasets. Each loader function
// returns a *tabgo.DataFrame ready for use in structure learning, parameter
// estimation, or inference examples.
package datasets

import (
	_ "embed"
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

//go:embed data/asia.csv
var asiaData string

//go:embed data/alarm.csv
var alarmData string

//go:embed data/sachs.csv
var sachsData string

//go:embed data/cancer.csv
var cancerData string

//go:embed data/student.csv
var studentData string

//go:embed data/sprinkler.csv
var sprinklerData string

//go:embed data/survey.csv
var surveyData string

//go:embed data/titanic.csv
var titanicData string

// Asia returns the Asia (Lauritzen-Spiegelhalter) network dataset.
// 1000 rows, 8 binary columns: asia, tub, smoke, lung, bronc, either, xray, dysp.
func Asia() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(asiaData)
}

// Alarm returns the Alarm (Burglary) network dataset.
// 1000 rows, 5 binary columns: Burglary, Earthquake, Alarm, JohnCalls, MaryCalls.
func Alarm() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(alarmData)
}

// Sachs returns the Sachs protein signaling network dataset.
// 500 rows, 11 columns discretized to 3 levels (0=low, 1=medium, 2=high):
// Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk.
func Sachs() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(sachsData)
}

// Cancer returns the Cancer network dataset.
// 1000 rows, 5 binary columns: Pollution, Smoker, Cancer, Xray, Dyspnoea.
func Cancer() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(cancerData)
}

// Student returns the Student network dataset.
// 1000 rows, 5 columns: D (binary), I (binary), G (ternary 0/1/2), L (binary), S (binary).
func Student() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(studentData)
}

// Sprinkler returns the Water Sprinkler network dataset.
// 1000 rows, 4 binary columns: Cloudy, Sprinkler, Rain, WetGrass.
func Sprinkler() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(sprinklerData)
}

// Survey returns the Survey network dataset.
// 500 rows, 5 categorical columns: Age, Education, Occupation, Residence, Transportation.
func Survey() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(surveyData)
}

// Titanic returns the Titanic survival dataset.
// 800 rows, 4 categorical columns: Class, Sex, Age, Survived.
func Titanic() (*tabgo.DataFrame, error) {
	return tabgo.ReadCSVFromString(titanicData)
}

// registry maps dataset names to loader functions.
var registry = map[string]func() (*tabgo.DataFrame, error){
	"asia":      Asia,
	"alarm":     Alarm,
	"sachs":     Sachs,
	"cancer":    Cancer,
	"student":   Student,
	"sprinkler": Sprinkler,
	"survey":    Survey,
	"titanic":   Titanic,
}

// List returns the names of all available datasets in sorted order.
func List() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Load loads a dataset by name. The name is case-sensitive and must match
// one of the names returned by List.
func Load(name string) (*tabgo.DataFrame, error) {
	fn, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("datasets: unknown dataset %q; available: %v", name, List())
	}
	return fn()
}
