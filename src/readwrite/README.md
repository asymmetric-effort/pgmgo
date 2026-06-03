# readwrite

Package `readwrite` provides readers and writers for probabilistic model file formats.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/readwrite`

## Supported Formats

| Format | Read | Write | Description |
|--------|------|-------|-------------|
| BIF | `ReadBIF` | `WriteBIF` | Bayesian Interchange Format |
| XMLBIF | `ReadXMLBIF` | `WriteXMLBIF` | XML-based BIF format |
| NET | `ReadNET` | `WriteNET` | Hugin NET format |
| UAI | `ReadUAI` | `WriteUAI` | UAI competition format |
| XDSL | `ReadXDSL` | `WriteXDSL` | GeNIe XDSL format |
| PomdpX | `ReadPomdpX` | - | POMDP-X format (BN subset) |
| XBN | `ReadXBN` | - | Microsoft XBN format |
| CSV | `ReadCSVStructure` / `ReadCSVWithCPDs` | `WriteCSV` | Edge-list CSV and CPD CSV |
| JSON | `ReadJSON` | `WriteJSON` | pgmgo JSON format with nodes, edges, states, CPDs |
| XML | `ReadXML` | `WriteXML` | pgmgo-native XML format |
| Parquet | - | - | Not yet implemented (placeholder) |
| XLSX | - | - | Not yet implemented (placeholder) |

All reader functions accept `io.Reader` and all writer functions accept `io.Writer`, making them composable with files, buffers, and network streams.

## Usage Examples

### BIF Format

```go
import (
    "os"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
)

// Read a BIF file
f, _ := os.Open("alarm.bif")
bn, _ := readwrite.ReadBIF(f)
f.Close()

// Write a BIF file
out, _ := os.Create("output.bif")
readwrite.WriteBIF(out, bn)
out.Close()
```

### JSON Format

```go
f, _ := os.Open("model.json")
bn, _ := readwrite.ReadJSON(f)
f.Close()

out, _ := os.Create("model.json")
readwrite.WriteJSON(out, bn)
out.Close()
```

### CSV Edge-List

```go
// CSV with "from,to" header
f, _ := os.Open("edges.csv")
bn, _ := readwrite.ReadCSVStructure(f) // structure only, no CPDs
f.Close()

// CSV with CPDs
f2, _ := os.Open("model.csv")
bn2, _ := readwrite.ReadCSVWithCPDs(f2)
f2.Close()
```

### UAI Format

```go
f, _ := os.Open("model.uai")
bn, _ := readwrite.ReadUAI(f)
f.Close()

out, _ := os.Create("model.uai")
readwrite.WriteUAI(out, bn)
out.Close()
```

### XMLBIF Format

```go
f, _ := os.Open("model.xmlbif")
bn, _ := readwrite.ReadXMLBIF(f)
f.Close()
```

### NET Format (Hugin)

```go
f, _ := os.Open("model.net")
bn, _ := readwrite.ReadNET(f)
f.Close()
```

### XDSL Format (GeNIe)

```go
f, _ := os.Open("model.xdsl")
bn, _ := readwrite.ReadXDSL(f)
f.Close()
```

### XBN Format (Microsoft)

```go
f, _ := os.Open("model.xbn")
bn, _ := readwrite.ReadXBN(f)
f.Close()
```

### pgmgo Native XML

```go
f, _ := os.Open("model.xml")
bn, _ := readwrite.ReadXML(f)
f.Close()

out, _ := os.Create("model.xml")
readwrite.WriteXML(out, bn)
out.Close()
```
