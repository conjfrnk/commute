// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "CommuteBar",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "CommuteBar", targets: ["CommuteBar"])],
    targets: [.executableTarget(name: "CommuteBar")]
)
