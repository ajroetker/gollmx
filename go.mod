module github.com/ajroetker/gollmx

go 1.26

require (
	github.com/ajroetker/go-highway v0.0.12
	github.com/gomlx/go-huggingface v0.3.2-0.20260125064416-b0f56ca7fbef
	github.com/gomlx/gomlx v0.26.1-0.20260224065554-7df01ab5c618
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	k8s.io/klog/v2 v2.140.0
)

replace github.com/gomlx/go-huggingface => ../go-huggingface

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/eliben/go-sentencepiece v0.7.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/go-xla v0.1.5-0.20260219173412-338774b2e7a7 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	golang.org/x/exp v0.0.0-20260218203240-3dfff04db8fa // indirect
	golang.org/x/image v0.34.0 // indirect
	golang.org/x/sys v0.41.0 // indirect
	golang.org/x/term v0.40.0 // indirect
	golang.org/x/text v0.34.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
