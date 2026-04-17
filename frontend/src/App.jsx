import './index.css'
import Navbar from './components/Navbar/Navbar'
import Hero from './components/Hero/Hero'
import Overview from './components/Overview/Overview'
import Demo from './components/Demo/Demo'
import MathSection from './components/MathSection/MathSection'
import Pipeline from './components/Pipeline/Pipeline'
import Modules from './components/Modules/Modules'
import HowToRun from './components/HowToRun/HowToRun'
import Validation from './components/Validation/Validation'
import Footer from './components/Footer/Footer'

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Overview />
        <Demo />
        <MathSection />
        <Pipeline />
        <Modules />
        <HowToRun />
        <Validation />
      </main>
      <Footer />
    </>
  )
}
