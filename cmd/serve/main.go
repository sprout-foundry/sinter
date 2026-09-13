// Command sinter-serve runs the OpenAI-compatible HTTP server from
// llm/openaisserver on top of a local sinter model. Minimal wiring example
// for serving on-device inference to local clients.
//
//	go run -tags ggml ./cmd/serve -model ~/.cache/sprout/models/qwen3-0.6b-4bit
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/all"
	"github.com/sprout-foundry/sinter/llm/openaisserver"
)

func main() {
	modelDir := flag.String("model", "", "path to a model directory (safetensors + config.json + tokenizer.json)")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	maxTokensCap := flag.Int("max-tokens-cap", 1024, "per-request max_tokens cap (0 = no cap)")
	flag.Parse()

	if *modelDir == "" {
		flag.Usage()
		os.Exit(2)
	}
	dir := *modelDir
	if strings.HasPrefix(dir, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			dir = filepath.Join(home, strings.TrimPrefix(dir, "~"))
		}
	}

	model, err := llm.NewModel(dir)
	if err != nil {
		log.Fatalf("sinter-serve: load: %v", err)
	}
	defer model.Close()

	srv := openaisserver.New(model, filepath.Base(dir), *maxTokensCap)
	httpSrv := &http.Server{Addr: *addr, Handler: srv.Handler()}

	go func() {
		log.Printf("sinter-serve: listening on %s (model %s)", *addr, filepath.Base(dir))
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("sinter-serve: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(ctx)
}
