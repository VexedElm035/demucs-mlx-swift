import DemucsMLX
import Foundation
import Testing

@Test("Registry exposes htdemucs")
func registryContainsDefaultModel() {
    #expect(listAvailableDemucsModels().contains("htdemucs"))
}

@Test("Parameter validation rejects invalid overlap")
func invalidOverlapRejected() {
    #expect(throws: DemucsError.self) {
        _ = try DemucsSeparator(
            modelName: "htdemucs",
            parameters: DemucsSeparationParameters(overlap: 1.1)
        )
    }
}
