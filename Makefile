.PHONY: build debug test clean

build:
	swift build -c release --disable-sandbox
	./scripts/build_mlx_metallib.sh release

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test:
	swift test --disable-sandbox

clean:
	swift package clean
