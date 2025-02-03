import { Container, Heading } from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { DeepChatComponent } from "../../components/Chat/DeepChatComponent";


export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
})

function Dashboard() {

  return (
    <>
	  <Container maxW="full">
      <Heading size="lg" textAlign={{ base: "center", md: "left" }} pt={12}>
        Deep Chat
      </Heading>
      <DeepChatComponent />
    </Container>
    </>
  )
}
